"""Read-only iPhone screenshot source via pymobiledevice3.

This module never sends touches, HID events, or any other input. It only
lists the device, pairs, mounts developer services when needed, and grabs
the screen.

On iOS 17.4+ (the usual case on current iPhones) pymobiledevice3 opens a
no-root in-process RSD tunnel automatically. iOS 17.0-17.3.1 on Windows
needs a privileged ``tunneld``; that is reported clearly if it comes up.
"""

from __future__ import annotations

import asyncio
import threading
import time
from word_solver.capture.config import DEFAULT_CAPTURE_CONFIG, CaptureConfig
from word_solver.capture.sources import (
    CapturedFrame,
    CaptureError,
    DisconnectedError,
    SourceStatus,
    decode_png,
)


class IPhoneCapture:
    """Persistent USB screenshot session.

    Capture calls are synchronous so the rest of the app does not need asyncio.
    A dedicated event loop on a background thread owns the RSD tunnel (one
    tunnel per process).
    """

    def __init__(self, config: CaptureConfig = DEFAULT_CAPTURE_CONFIG, udid: str | None = None) -> None:
        self.config = config
        self.udid = udid
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="iphone-capture", daemon=True)
        self._thread.start()
        self._lock = threading.Lock()
        self._provider = None
        self._tunnel = None
        self._lockdown = None
        self._dvt = None
        self._screenshot = None
        self._info: dict = {}

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _call(self, coro, timeout: float | None = None):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout or self.config.capture_timeout_seconds)

    # ----- public --------------------------------------------------------

    def status(self) -> SourceStatus:
        try:
            return self._call(self._status(), timeout=8)
        except DisconnectedError as exc:
            return SourceStatus(kind="iphone", connected=False, message=str(exc))
        except Exception as exc:
            return SourceStatus(kind="iphone", connected=False, message=_friendly(exc))

    def capture(self) -> CapturedFrame:
        start = time.perf_counter()
        try:
            png = self._call(self._take_png())
        except DisconnectedError:
            self._reset_session()
            raise
        except Exception as exc:
            self._reset_session()
            raise CaptureError(_friendly(exc)) from exc
        image = decode_png(png)
        elapsed = (time.perf_counter() - start) * 1000
        name = self._info.get("name") or "iPhone"
        return CapturedFrame(image, png, elapsed, name, extra=dict(self._info))

    def pair(self) -> SourceStatus:
        return self._call(self._pair(), timeout=self.config.pair_timeout_seconds)

    def prepare_developer(self) -> str:
        """Enable Developer Mode if needed and mount the Developer Disk Image.

        Mount is required once per device boot. Returns a human-readable summary.
        """
        return self._call(self._prepare_developer(), timeout=180)

    def close(self) -> None:
        """Drop the current screenshot session so the next capture reconnects."""
        self._reset_session()

    def shutdown(self) -> None:
        """Close the session and stop the background asyncio thread."""
        self.close()
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=3)

    # ----- async internals ----------------------------------------------

    async def _usb(self):
        from pymobiledevice3.usbmux import list_devices

        devices = [d for d in await list_devices() if d.is_usb]
        if self.udid:
            devices = [d for d in devices if d.matches_udid(self.udid)]
        if not devices:
            raise DisconnectedError(
                "No iPhone on USB. Use a data cable (not charge-only), unlock the phone, "
                "and confirm Apple Mobile Device Service is running."
            )
        return devices[0]

    async def _open_lockdown(self, autopair: bool = False):
        from pymobiledevice3.lockdown import create_using_usbmux

        device = await self._usb()
        return await create_using_usbmux(
            serial=device.serial,
            autopair=autopair,
            connection_type="USB",
            pair_timeout=self.config.pair_timeout_seconds,
        )

    async def _status(self) -> SourceStatus:
        from pymobiledevice3.exceptions import NotPairedError, SessionActiveError

        device = await self._usb()
        try:
            lockdown = await self._open_lockdown(autopair=False)
        except NotPairedError:
            return SourceStatus(
                kind="iphone",
                connected=True,
                message="iPhone found but not paired. Unlock it, then run: python -m word_solver.capture pair",
                udid=device.serial,
                paired=False,
            )
        except SessionActiveError:
            return SourceStatus(
                kind="iphone",
                connected=True,
                message="iPhone is connected, but another lockdown session is using it. Close iTunes or other Apple tools and retry.",
                udid=device.serial,
            )
        try:
            paired = bool(lockdown.paired)
            name = lockdown.short_info.get("DeviceName") or lockdown.display_name
            ios = lockdown.product_version
            developer_mode = None
            ddi_mounted = None
            if paired:
                try:
                    developer_mode = await lockdown.get_developer_mode_status()
                except Exception:
                    developer_mode = None
                ddi_mounted = await _ddi_mounted(lockdown)
            self._info = {"udid": device.serial, "name": name, "ios": ios}
            return SourceStatus(
                kind="iphone",
                connected=True,
                message=_status_message(paired, developer_mode, ddi_mounted, ios),
                udid=device.serial,
                name=name,
                ios=ios,
                paired=paired,
                developer_mode=developer_mode,
                ddi_mounted=ddi_mounted,
            )
        finally:
            await _close_lockdown(lockdown)

    async def _pair(self) -> SourceStatus:
        from pymobiledevice3.exceptions import PairingDialogResponsePendingError, PasswordRequiredError

        lockdown = await self._open_lockdown(autopair=False)
        try:
            if not lockdown.paired:
                await lockdown.pair(timeout=self.config.pair_timeout_seconds)
        except PasswordRequiredError as exc:
            raise CaptureError("Unlock the iPhone and try pairing again.") from exc
        except PairingDialogResponsePendingError as exc:
            raise CaptureError("Tap Trust on the iPhone, then run pair again.") from exc
        finally:
            await _close_lockdown(lockdown)
        return await self._status()

    async def _prepare_developer(self) -> str:
        lockdown = await self._open_lockdown(autopair=True)
        notes = []
        try:
            try:
                enabled = await lockdown.get_developer_mode_status()
            except Exception:
                enabled = None
            if enabled is False:
                from pymobiledevice3.services.amfi import AmfiService

                notes.append("Requesting Developer Mode (the iPhone will reboot). Confirm on the device.")
                await AmfiService(lockdown).enable_developer_mode(enable_post_restart=True)
                await _close_lockdown(lockdown)
                lockdown = await self._open_lockdown(autopair=True)
                notes.append("Developer Mode enabled.")
            elif enabled:
                notes.append("Developer Mode already on.")
            else:
                notes.append("Could not read Developer Mode status (older iOS may not need it).")

            try:
                await _auto_mount(lockdown)
                notes.append("Developer Disk Image mounted.")
            except ImportError:
                notes.append(
                    "Developer Disk Image mounter is unavailable because pyimg4/pylzss did not install "
                    "(Python 3.14 needs Microsoft C++ Build Tools). If screenshots fail, install Build Tools "
                    "and reinstall pymobiledevice3, or mount the DDI from Xcode once."
                )
            except Exception as exc:
                notes.append(f"DDI mount: {_friendly(exc)}")
            return " ".join(notes)
        finally:
            await _close_lockdown(lockdown)

    async def _take_png(self) -> bytes:
        await self._ensure_session()
        assert self._screenshot is not None
        return await self._screenshot.get_screenshot()

    async def _ensure_session(self) -> None:
        if self._screenshot is not None:
            return
        from packaging.version import Version
        from pymobiledevice3.lockdown import create_using_usbmux
        from pymobiledevice3.services.dvt.instruments.dvt_provider import DvtProvider
        from pymobiledevice3.services.dvt.instruments.screenshot import Screenshot

        device = await self._usb()
        lockdown = await create_using_usbmux(serial=device.serial, autopair=True, connection_type="USB")
        self._info = {
            "udid": device.serial,
            "name": lockdown.short_info.get("DeviceName") or lockdown.display_name,
            "ios": lockdown.product_version,
        }
        ios = Version(lockdown.product_version)
        if ios >= Version("17.0"):
            from pymobiledevice3.remote.rsd_tunnel import PreferredRsdTunnel

            # One lockdown session at a time on Windows: close before opening the RSD tunnel.
            await _close_lockdown(lockdown)
            self._tunnel = PreferredRsdTunnel(serial=device.serial, autopair=True)
            provider = await self._tunnel.aopen()
        else:
            self._lockdown = lockdown
            provider = lockdown
        self._provider = provider
        self._dvt = DvtProvider(provider)
        await self._dvt.__aenter__()
        self._screenshot = Screenshot(self._dvt)
        await self._screenshot.__aenter__()

    async def _aclose_session(self) -> None:
        for obj, name in ((self._screenshot, "__aexit__"), (self._dvt, "__aexit__")):
            if obj is None:
                continue
            try:
                await obj.__aexit__(None, None, None)
            except Exception:
                pass
        if self._tunnel is not None:
            try:
                await self._tunnel.aclose()
            except Exception:
                pass
        await _close_lockdown(self._lockdown)
        self._screenshot = None
        self._dvt = None
        self._tunnel = None
        self._lockdown = None
        self._provider = None

    def _reset_session(self) -> None:
        try:
            self._call(self._aclose_session(), timeout=5)
        except Exception:
            self._screenshot = None
            self._dvt = None
            self._tunnel = None
            self._lockdown = None
            self._provider = None


async def _close_lockdown(lockdown) -> None:
    if lockdown is None:
        return
    try:
        await lockdown.close()
    except Exception:
        pass


async def _ddi_mounted(lockdown) -> bool | None:
    try:
        from pymobiledevice3.services.mobile_image_mounter import MobileImageMounterService
    except ImportError:
        return None
    try:
        async with MobileImageMounterService(lockdown) as mounter:
            images = await mounter.list_images()
        return bool(images)
    except Exception:
        return None


async def _auto_mount(lockdown) -> None:
    from pymobiledevice3.exceptions import AlreadyMountedError
    from pymobiledevice3.services.mobile_image_mounter import auto_mount

    try:
        await auto_mount(lockdown)
    except AlreadyMountedError:
        return


def _status_message(paired, developer_mode, ddi_mounted, ios) -> str:
    if paired is False:
        return "Connected but not paired. Unlock the iPhone and run: python -m word_solver.capture pair"
    parts = ["iPhone connected"]
    if ios:
        parts.append(f"iOS {ios}")
    if developer_mode is False:
        parts.append("Developer Mode is OFF (Settings -> Privacy & Security -> Developer Mode)")
    if ddi_mounted is False:
        parts.append("Developer Disk Image not mounted (python -m word_solver.capture prepare)")
    return ". ".join(parts) + "."


def _friendly(exc: BaseException) -> str:
    name = type(exc).__name__
    text = str(exc) or name
    mapping = {
        "SessionActiveError": "iPhone is connected, but another lockdown session is using it. Close iTunes or other Apple tools and retry.",
        "NoDeviceConnectedError": "No iPhone on USB.",
        "NotPairedError": "iPhone is not paired. Unlock it and tap Trust.",
        "PairingDialogResponsePendingError": "Tap Trust on the iPhone.",
        "PasswordRequiredError": "Unlock the iPhone.",
        "InvalidServiceError": "Developer services are not available. Enable Developer Mode and mount the DDI.",
        "UserspaceTunnelUnavailableError": "Could not open the iOS 17+ tunnel. For iOS 17.0-17.3.1 on Windows, run tunneld as Administrator.",
        "DeviceHasPasscodeSetError": "Turn Developer Mode on manually: Settings -> Privacy & Security -> Developer Mode.",
        "TunneldConnectionError": "tunneld is not running. Needed only for iOS 17.0-17.3.1 on Windows.",
        "AlreadyMountedError": "Developer Disk Image is already mounted.",
    }
    return mapping.get(name, text)

"""
SSL Certificate Verification Bypass Patch
This module patches SSL and WebSocket to bypass certificate verification.
Should only be used in development/testing environments.
"""
import ssl
import websocket

# Disable SSL certificate verification globally
ssl._create_default_https_context = ssl._create_unverified_context

# Monkey patch WebSocketApp.run_forever to force sslopt with cert verification disabled
if hasattr(websocket, 'WebSocketApp'):
    _orig_run_forever = websocket.WebSocketApp.run_forever

    def _patched_run_forever(self, *args, **kwargs):
        if 'sslopt' not in kwargs or kwargs['sslopt'] is None:
            kwargs['sslopt'] = {"cert_reqs": ssl.CERT_NONE}
        else:
            kwargs['sslopt']['cert_reqs'] = ssl.CERT_NONE
        print("[SSL Patch] WebSocketApp.run_forever: sslopt=", kwargs['sslopt'])
        return _orig_run_forever(self, *args, **kwargs)

    websocket.WebSocketApp.run_forever = _patched_run_forever
    print("[SSL Patch] websocket.WebSocketApp.run_forever patched to ignore SSL certificate verification.")

# Patch WebSocket class if available
if hasattr(websocket, 'WebSocket'):
    _orig_websocket_init = websocket.WebSocket.__init__

    def _patched_websocket_init(self, *args, **kwargs):
        _orig_websocket_init(self, *args, **kwargs)
        # Set sslopt to ignore certificate verification
        if not hasattr(self, 'sslopt') or self.sslopt is None:
            self.sslopt = {"cert_reqs": ssl.CERT_NONE}
        else:
            self.sslopt['cert_reqs'] = ssl.CERT_NONE

    websocket.WebSocket.__init__ = _patched_websocket_init
    print("[SSL Patch] websocket.WebSocket patched to ignore SSL certificate verification.")

print("[SSL Patch] SSL certificate verification has been disabled globally.")
print("[SSL Patch] WARNING: This should only be used in development/testing environments!")

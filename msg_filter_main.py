import sys

msg = sys.stdin.read()
prefixes = [
    "\U0001F9F9 ",
    "\U0001F680 ",
    "\u011F\u0178\u00A7\u00B9 ",
    "\u011F\u0178\u0161\u20AC ",
]
for prefix in prefixes:
    if msg.startswith(prefix):
        msg = msg[len(prefix):]
        break
sys.stdout.write(msg)

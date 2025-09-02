import datetime


def time_now():
    return datetime.datetime.now()


def time_past(since, millis=True):
    ut = str(time_now() - since)

    # remove leading zeroes and colons
    for i, c in enumerate(ut):
        if c not in [":", "0"]:
            ut = ut[i:]
            break

    parts = ut.split(".")
    if millis and len(parts) > 1:
        return f"{parts[0]}.{parts[1][:2]}"
    else:
        return parts[0]


class TimingLogger(object):
    def __init__(self, name=""):
        self.name = name
        self.start_time = time_now()
        self.num_logs = 0

    def report(self, msg="No message", printer=print):
        self.num_logs += 1
        msg = f"{self.name} {self.num_logs} | {time_past(self.start_time, millis=True)}: {msg}"
        printer(msg)
        return msg

    def time_past_seconds(self):
        return (time_now() - self.start_time).total_seconds()

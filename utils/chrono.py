from collections import defaultdict
from timeit import default_timer as timer


class Chronometer:
    def __init__(self, identifier):
        self.identifier = identifier

    def __enter__(self):
        self.start = timer()

    def __exit__(self, *args):
        #print("elapsed time ({}): {}".format(self.identifier, (timer() - self.start)))
        pass


class Chronostep:
    def __init__(self, identifier, inline=False):
        self.identifier = identifier
        self.inline = inline

    def __enter__(self):
        self.start = timer()
        print("{}...".format(self.identifier), end="")
        if not self.inline:
            print()

    def __exit__(self, *args):
        elapsed_seconds = round(timer() - self.start, 4)
        if self.inline:
            print("\r{}: done in {} seconds".format(self.identifier, elapsed_seconds))
        else:
            print("{}: done in {} seconds".format(self.identifier, elapsed_seconds))


class Chronoloop:
    def __init__(self):
        self.starting_times = {}
        self.times = defaultdict(lambda: 0)

    def __enter__(self):
        self.start_time = timer()
        return self

    def start(self, identifier):
        self.starting_times[identifier] = timer()

    def stop(self, identifier):
        self.times[identifier] += timer() - self.starting_times[identifier]
        del self.starting_times[identifier]

    def stop_all(self):
        for identifier in list(self.starting_times):
            self.stop(identifier)

    def stop_all_and_start(self, identifier):
        self.stop_all()
        self.start(identifier)

    def __exit__(self, *args):
        self.stop_all()
        total_seconds = round(timer() - self.start_time, 4)
        print("total: {} seconds, detail: {}".format(total_seconds, dict(self.times)))

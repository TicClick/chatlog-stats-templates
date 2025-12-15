#!/usr/bin/env python3

import argparse
import collections
import dataclasses
import datetime
import gzip
import html
import json
import os
import pathlib
import random
import re
import time
import tomllib
import sys

from jinja2 import Environment, FileSystemLoader
import ossapi


URLS_PATTERN = re.compile(r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)")


def debug(s: str):
    print(s, file=sys.stderr)


@dataclasses.dataclass
class FileConfig:
    template_dir: str
    logs_path: str
    generate_to: str
    save_as: str

@dataclasses.dataclass
class DateConfig:
    year: str
    month: str

@dataclasses.dataclass
class GeneratorConfig:
    files: FileConfig
    date: DateConfig


def get_filter_mask(year=r"\d+", month=r"\d+"):
    if year == "current":
        year = datetime.datetime.now().strftime("%Y")
    if month == "current":
        month = datetime.datetime.now().strftime("%m")
    if month == "previous" or year == "previous":
        now = datetime.datetime.now()
        month_number = now.month - 1
        year_number = now.year
        if month_number < 0:
            month_number = month_number % 12
            year_number = year_number - 1
        month_number += 1
        prev = datetime.datetime(year_number, month_number, 15)
        year = prev.strftime("%Y")
        month = prev.strftime("%m")

    return re.compile(r"(%s)\-(%s)\-\d+\.\w+" % (year, month), re.IGNORECASE)


@dataclasses.dataclass
class MessageType:
    REGULAR = 0
    ACTION = 1
    MODE_CHANGE = 2


@dataclasses.dataclass
class ChatMessage:
    time: datetime.time
    username: str
    text: str
    line_type: int

    @classmethod
    def parse(cls, line: str):
        if len(line) < 7 or not ('0' <= line[0] and line[0] <= '9'):  # "HH:MM " + line contents
            return None

        contents = line.split()
        message_time = datetime.time(*map(int, contents[0].split(":")))

        match contents[1][0]:
            # 16:32:01  * -Aki_Minoriko is listening to [https://osu.ppy.sh/beatmapsets/1712992#/3500223 S3RL feat. Sara - Dopamine]
            case "*":
                return cls(
                    line_type=MessageType.ACTION,
                    time=message_time,
                    username=contents[2],
                    text=" ".join(contents[3:]),
                )

            # 16:37:03 < _dopamine> юкр.
            # 16:39:11 <@Kobold84> Юкр.
            case "<":
                return cls(
                    line_type=MessageType.REGULAR,
                    time=message_time,
                    username=contents[2].rstrip(">"),
                    text=" ".join(contents[3:]),
                )
    
            # 16:45:29 -!- mode/#russian [+o terho] by BanchoBot
            case "-":
                return cls(
                    line_type=MessageType.MODE_CHANGE,
                    time=message_time,
                    username=contents[4].rstrip("]"),
                    text="",
                )
            case d:
                raise RuntimeError(f"Failed to parse line {line!r} -- unknown discriminator {d!r} (second element: {contents[1]!r})")

    def parse_urls(self):
        return URLS_PATTERN.findall(self.text)


@dataclasses.dataclass
class User:
    user_id: int
    username: str
    message_count: int
    random_quote: str


@dataclasses.dataclass
class Url:
    address: str
    count: int
    last_used_username: str


class APIClient:
    _UID_CACHE_PATH = os.path.join(os.path.dirname(__file__), ".uid-cache.toml")

    def __init__(self, credentials_path):
        self._cache = {}

        with open(credentials_path, "rb") as fd:
            data = tomllib.load(fd)
            try:
                with open(self._UID_CACHE_PATH, "rb") as fd:
                    self._cache = tomllib.load(fd)
            except OSError:
                pass

        self.api = ossapi.Ossapi(
            client_id=data["client_id"],
            client_secret=data["client_secret"],
        )

    def uid(self, username):
        username = username.lower().strip().replace(" ", "_")
        if username not in self._cache:
            try:
                data = self.api.user(f"@{username}")
                self._cache[username] = data.id
                try:
                    with open(self._UID_CACHE_PATH, "a") as fd:
                        fd.write(f"\"{username}\" = {data.id}\n")
                except OSError:
                    pass  # don't sweat over it
            except Exception as e:
                debug(e)
        
        return self._cache.get(username)


# Needed for two reasons: 1) avoid falling back to external tools, and 2) avoid keeping logs in memory.
# See https://en.wikipedia.org/wiki/Reservoir_sampling for details.
class ReservoirSampler:
    def __init__(self):
        self._storage = {}
        self._counter = collections.Counter()

    def consider(self, k, v):
        self._counter[k] += 1
        if k not in self._storage:
            self._storage[k] = v
        else:
            curr_idx = self._counter[k]
            chosen_element_idx = random.randint(1, curr_idx)
            if chosen_element_idx == 1:
                self._storage[k] = v

    def get(self, k):
        return self._storage.get(k)


class Main:
    def __init__(self, config_path: str, channel: str, api_credentials_path: str):
        with open(config_path, "rb") as fd:
            data = tomllib.load(fd)
            self.config = GeneratorConfig(
                files=FileConfig(**data["files"]),
                date=DateConfig(**data["date"])
            )

        self.channel_name = channel
        self.api = APIClient(credentials_path=api_credentials_path)

        self.template = Environment(
            loader=FileSystemLoader(self.config.files.template_dir)
        ).get_template('template.html')

        # per user statistics
        self.user_messages: collections.Counter[str] = collections.Counter()
        self.user_question: collections.Counter[str] = collections.Counter()
        self.user_exclamation: collections.Counter[str] = collections.Counter()
        self.user_actions: collections.Counter[str] = collections.Counter()
        self.user_givemodes: collections.Counter[str] = collections.Counter()

        self._cache_user_messages = ReservoirSampler()

        self.activity_graph = [0]*24

        # Per URL statistics
        self.url_count: collections.Counter[str] = collections.Counter()
        self.last_url_usage: dict[str, str] = collections.defaultdict(str)

        filter_mask = get_filter_mask(self.config.date.year, self.config.date.month)
        channel_logs_dir = os.path.join(self.config.files.logs_path, self.channel_name)
        matching_filenames = filter(
            lambda filename: os.path.isfile(os.path.join(channel_logs_dir, filename)) and filter_mask.search(filename),
            os.listdir(channel_logs_dir)
        )
        self.matching_paths = list(
            os.path.join(channel_logs_dir, filename)
            for filename in matching_filenames
        )

        debug(f"Found {len(self.matching_paths)} matching log(s) in {channel_logs_dir}")


    def bulk_lines(self):
        now = time.time()
        for filepath in self.matching_paths:
            filepath = pathlib.Path(filepath)
            file_ctx = (
                gzip.open(filepath, "rb")
                if filepath.suffix == ".gz" else
                open(filepath, "r")
            )
            with file_ctx as fd:
                for line in fd:
                    self.one_line(line.strip())

        elapsed = time.time() - now
        debug(f"Parsed {len(self.matching_paths)} log(s) in {elapsed:.3}s")

    def one_line(self, line):
        message = ChatMessage.parse(line)
        if message is None:
            debug(f"Skipped line: {line!r}")
            return None

        u = message.username
        match message.line_type:
            case MessageType.MODE_CHANGE:
                self.user_givemodes[u] += 1
            
            case MessageType.REGULAR | MessageType.ACTION:
                self.activity_graph[message.time.hour] += 1
                self.user_messages[u] += 1

                if message.line_type == MessageType.ACTION:
                    self.user_actions[u] += 1

                self._cache_user_messages.consider(u, message.text)

                for url in message.parse_urls():
                    self.url_count[url] += 1
                    self.last_url_usage[url] = u

                if "!" in message.text:
                    self.user_exclamation[u] += 1

                if "?" in message.text:
                    self.user_question[u] += 1

    def save_page(self):
        users_messages_desc = self.user_messages.most_common()

        capped_top25_len = min(25, len(users_messages_desc))
        top25 = users_messages_desc[:capped_top25_len]

        capped_top35_len = min(35, len(users_messages_desc))
        runner_ups = users_messages_desc[capped_top25_len:capped_top35_len]

        most_active: list[User] = []
        for (username, message_count) in top25:
            debug(f"Fething profile data and random quote for: {username}")
            uid = self.api.uid(username)
            random_quote = self._cache_user_messages.get(username)

            most_active.append(
                User(
                    user_id=uid,
                    username=username,
                    message_count=message_count,
                    random_quote=html.escape(random_quote),
                )
            )

        # graph percentage calculation
        sum = 0
        for i in self.activity_graph:
            sum += i
        sum = sum/240.0
        for i in range(0, 24):
            self.activity_graph[i] = int(self.activity_graph[i])

        being = {
            "screaming": self.user_exclamation.most_common(2),
            "asking": self.user_question.most_common(2),
            "telling": self.user_actions.most_common(2),
            "modding": self.user_givemodes.most_common(2),
        }

        top10_urls = [
            Url(
                address=item[0],
                count=item[1],
                last_used_username=self.last_url_usage[item[0]]
            )
            for item in self.url_count.most_common(10)
        ]

        total_num = [
            self.user_question.total(),
            self.user_exclamation.total(),
            self.user_actions.total(),
            self.user_givemodes.total(),
        ]

        debug(f"Rendering template file for {self.channel_name}")
        output_from_parsed_template = self.template.render(
            name=self.channel_name,
            most_active=most_active,
            runner_ups=runner_ups,
            being=being,
            urls_used=top10_urls,
            total=total_num,
            activity_graph=self.activity_graph
        )

        os.makedirs(self.config.files.generate_to, exist_ok=True)
        
        template_output_path = os.path.join(
            self.config.files.generate_to,
            self.config.files.save_as % self.channel_name
        )
        with open(template_output_path, "wb") as fh:
           fh.write(output_from_parsed_template.encode("utf-8"))

        json_output_path = os.path.join(
            self.config.files.generate_to,
            f"{self.channel_name}.json"
        )
        with open(json_output_path, "w") as fh:
            data = {
                "name": self.channel_name,
                "total_numbers": total_num,
                "most_active": list(map(
                    lambda user: (user.username, user.message_count),
                    most_active
                )),
                "runner_ups": runner_ups,
                "being": being,
                "most_used_urls": list(map(
                    lambda url: (url.address, url.count),
                    top10_urls
                )),
                "activity_graph": self.activity_graph
            }
            json.dump(data, fh)


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the .toml config file", required=True)
    parser.add_argument("--channel", help="IRC channel name", required=True)
    parser.add_argument(
        "--api-credentials", help="Path to the TOML file with osu! API v2 credentials (client_id/client_secret)",
        required=True
    )

    args = parser.parse_args(sys.argv[1:])

    a = Main(
        config_path=args.config,
        channel=args.channel,
        api_credentials_path=args.api_credentials
    )

    a.bulk_lines()
    a.save_page()

    debug("%s --- %s seconds ---" % (a.channel_name, (time.time() - start_time)))


if __name__ == "__main__":
    main()

import requests


class ThirtPartyEmotesParser:
    def __init__(self) -> None:
        pass

    def get_bttv_emotes(self, broadcaster_id):
        data = requests.get("https://api.betterttv.net/3/cached/emotes/global").json()
        global_emotes = [global_emote["code"] for global_emote in data]

        channel_emotes = []
        response = requests.get(
            f"https://api.betterttv.net/3/cached/users/twitch/{broadcaster_id}"
        )
        if response.status_code == 200:
            data = response.json()
            channel_emotes = [
                channel_emote["code"] for channel_emote in data["channelEmotes"]
            ]
        return global_emotes + channel_emotes

    def get_ffz_emotes(self, broadcaster_id):
        data = requests.get(
            "https://api.betterttv.net/3/cached/frankerfacez/emotes/global"
        ).json()
        global_emotes = [global_emote["code"] for global_emote in data]

        channel_emotes = []
        response = requests.get(
            f"https://api.betterttv.net/3/cached/frankerfacez/users/twitch/{broadcaster_id}"
        )
        if response.status_code == 200:
            data = response.json()
            channel_emotes = [channel_emote["code"] for channel_emote in data]
        return global_emotes + channel_emotes

    def get_7tv_emotes(self, broadcaster_id):
        data = requests.get("https://7tv.io/v3/emote-sets/global").json()
        global_emotes = [global_emote["name"] for global_emote in data["emotes"]]

        channel_emotes = []
        response = requests.get(f"https://7tv.io/v3/users/twitch/{broadcaster_id}")
        if response.status_code == 200:
            data = response.json()
            emote_set = data["emote_set"]
            channel_emotes = [
                channel_emote["name"] for channel_emote in emote_set["emotes"]
            ]
        return global_emotes + channel_emotes

    def get_all_channel_emotes(self, broadcaster_id):
        bttv_emotes = self.get_bttv_emotes(broadcaster_id)
        ffz_emotes = self.get_ffz_emotes(broadcaster_id)
        stv_emotes = self.get_7tv_emotes(broadcaster_id)
        all_emotes = list(set(bttv_emotes + ffz_emotes + stv_emotes))
        print(
            f"Successfully found {len(all_emotes)} emotes on channel of {broadcaster_id}"
        )
        return all_emotes

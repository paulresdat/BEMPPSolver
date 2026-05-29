from signalrcore.hub_connection_builder import HubConnectionBuilder, AuthHubConnection, BaseHubConnection
from typing import Optional

class Signalr(object):
    __connection: Optional[AuthHubConnection | BaseHubConnection] = None
    def __init__(self):
        pass

    def start(self, hub_url: str):
        self.__connection = HubConnectionBuilder()\
            .with_url(hub_url)\
            .with_automatic_reconnect({
                "type": "raw",
                "keep_alive_interval": 10,
                "reconnect_interval": 5,
                "max_attempts": 5
            }).build()
        self.__connection.start()

    def send_message(self, message: str):
        if self.__connection is not None:
            self.__connection.send("MessageFromClient", [message])

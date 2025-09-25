import json
import paho.mqtt.client as mqtt
from .config import MQTTConf

class MQTTClient:
    def __init__(self, conf: MQTTConf = MQTTConf()):
        self.conf = conf
        self.client = mqtt.Client()
        self.client.connect(conf.host, conf.port, keepalive=60)

    def publish_cmd(self, payload: dict):
        self.client.publish(self.conf.topic_cmd, json.dumps(payload), qos=1)

    def loop(self):
        self.client.loop(timeout=0.1)

    # Add subscribe/handlers if you want to react to sensors via MQTT directly

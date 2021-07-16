#!/usr/local/bin/python3

import argparse
import base64
import json

import tornado.ioloop
import tornado.web

from grl.utils.log import get_stdout_logger


class MainHandler(tornado.web.RequestHandler):
    
    def initialize(self):
        self.log = get_stdout_logger('link-prediction-server')

    def get(self):
        res = self.request.uri
        dec = receive_results(res[1:])
        self.log.info(res)
        self.log.info(dec)
        self.write(f"{res}\n\n{dec}")


def receive_results(x):
    return json.loads(base64.b64decode(x.encode()).decode())


def make_app():
    return tornado.web.Application([
        (r"/.*", MainHandler),
    ])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-p', '--port', type=int, default=8888)
    args, _ = p.parse_known_args()

    app = make_app()
    app.listen(args.port)
    tornado.ioloop.IOLoop.current().start()

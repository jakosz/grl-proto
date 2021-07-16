#!/usr/local/bin/python3

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
        self.log.info(res)


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()

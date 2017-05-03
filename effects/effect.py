#!/usr/bin/env python3

class Effect:
    def process(self, image):
        raise NotImplementedError


class Pipeline(Effect):
    def __init__(self):
        self.effects = []

    def add(self, effect):
        self.effects.append(effect)

    def add_list(self, effects):
        self.effects.extend(effects)

    def process(self, image):
        res = image
        for effect in self.effects:
            res = effect.process(res)
        return res
#!/usr/bin/python
# -*- coding: utf-8 -*-


class SetOfAssignments:
    def __init__(self, set_id, name, slug, assignments, is_bonus_set):
        self.set_id=set_id
        self.name=name                  # nazev setu
        self.slug=slug                  # to s pomlckama a bez diakritiky (bitva-o-brno)
        self.assignments=assignments    # id uloh, ktere patri do teto sady
        self.is_bonus_set=is_bonus_set  # jestli je to bonusovy set
        self.id_to_order={}             # mapuje id ulohy na jeji poradi, == slug_to_assign
        cnt=1
        for i in self.assignments:
            self.id_to_order[cnt] = i
            cnt += 1

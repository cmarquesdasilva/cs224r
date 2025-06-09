RESPONSE_TO_SCORE = {
    "Very Inaccurate": 1,
    "Moderately Inaccurate": 2,
    "Neither Accurate Nor Inaccurate": 3,
    "Moderately Accurate": 4,
    "Very Accurate": 5
}

FACET_TO_ITEMS = {
    # Neuroticism
    "anxiety": ['i1', 'i31', 'i61', 'i91'],
    "anger": ['i6', 'i36', 'i66', 'i96'],
    "depression": ['i11', 'i41', 'i71', 'i101'],
    "self_consciousness": ['i16', 'i46', 'i76', 'i106'],
    "immoderation": ['i21', 'i51', 'i81', 'i111'],
    "vulnerability": ['i26', 'i56', 'i86', 'i116'],

    # Extraversion
    "friendliness": ['i2', 'i32', 'i62', 'i92'],
    "gregariousness": ['i7', 'i37', 'i67', 'i97'],
    "assertiveness": ['i12', 'i42', 'i72', 'i102'],
    "active": ['i17', 'i47', 'i77', 'i107'],
    "excitement_seeking": ['i22', 'i52', 'i82', 'i112'],
    "cheerfulness": ['i27', 'i57', 'i87', 'i117'],

    # Openness
    "imagination": ['i3', 'i33', 'i63', 'i93'],
    "artistic_interests": ['i8', 'i38', 'i68', 'i98'],
    "emotionality": ['i13', 'i43', 'i73', 'i103'],
    "adventurousness": ['i18', 'i48', 'i78', 'i108'],
    "intellect": ['i23', 'i53', 'i83', 'i113'],
    "liberalism": ['i28', 'i58', 'i88', 'i118'],

    # Agreeableness
    "trust": ['i4', 'i34', 'i64', 'i94'],
    "morality": ['i9', 'i39', 'i69', 'i99'],
    "altruism": ['i14', 'i44', 'i74', 'i104'],
    "cooperation": ['i19', 'i49', 'i79', 'i109'],
    "modesty": ['i24', 'i54', 'i84', 'i114'],
    "sympathy": ['i29', 'i59', 'i89', 'i119'],

    # Conscientiousness
    "self_efficacy": ['i5', 'i35', 'i65', 'i95'],
    "orderliness": ['i10', 'i40', 'i70', 'i100'],
    "dutifulness": ['i15', 'i45', 'i75', 'i105'],
    "achievement_striving": ['i20', 'i50', 'i80', 'i110'],
    "self_discipline": ['i25', 'i55', 'i85', 'i115'],
    "cautiousness": ['i30', 'i60', 'i90', 'i120']
}

DOMAIN_TO_FACETS = {
    "neuroticism": [
        "anxiety",
        "anger",
        "depression",
        "self_consciousness",
        "immoderation",
        "vulnerability"
    ],
    "extraversion": [
        "friendliness",
        "gregariousness",
        "assertiveness",
        "active",
        "excitement_seeking",
        "cheerfulness"
    ],
    "openness": [
        "imagination",
        "artistic_interests",
        "emotionality",
        "adventurousness",
        "intellect",
        "liberalism"
    ],
    "agreeableness": [
        "trust",
        "morality",
        "altruism",
        "cooperation",
        "modesty",
        "sympathy"
    ],
    "conscientiousness": [
        "self_efficacy",
        "orderliness",
        "dutifulness",
        "achievement_striving",
        "self_discipline",
        "cautiousness"
    ]
}


COLUMN_INDEX_TO_KEY = {'i0': 1, 'i1': 1, 'i2': 1, 'i3': 1, 'i4': 1, 'i5': 1, 'i6': 1, 'i7': 1, 'i8': -1, 'i9': 1, 'i10': 1, 'i11': 1, 'i12': 1, 'i13': 1, 'i14': 1, 'i15': 1, 'i16': 1, 'i17': 1, 'i18': -1, 'i19': 1, 'i20': 1, 'i21': 1, 'i22': 1, 'i23': -1, 'i24': 1, 'i25': 1, 'i26': 1, 'i27': 1, 'i28': 1, 'i29': -1, 'i30': 1, 'i31': 1, 'i32': 1, 'i33': 1, 'i34': 1, 'i35': 1, 'i36': 1, 'i37': 1, 'i38': -1, 'i39': -1, 'i40': 1, 'i41': 1, 'i42': 1, 'i43': 1, 'i44': 1, 'i45': 1, 'i46': 1, 'i47': -1, 'i48': -1, 'i49': 1, 'i50': -1, 'i51': 1, 'i52': -1, 'i53': -1, 'i54': 1, 'i55': 1, 'i56': 1, 'i57': 1, 'i58': 1, 'i59': -1, 'i60': 1, 'i61': -1, 'i62': 1, 'i63': 1, 'i64': 1, 'i65': 1, 'i66': -1, 'i67': -1, 'i68': -1, 'i69': -1, 'i70': 1, 'i71': 1, 'i72': -1, 'i73': -1, 'i74': -1, 'i75': 1, 'i76': 1, 'i77': -1, 'i78': -1, 'i79': -1, 'i80': -1, 'i81': 1, 'i82': -1, 'i83': -1, 'i84': -1, 'i85': 1, 'i86': 1, 'i87': -1, 'i88': -1, 'i89': -1, 'i90': 1, 'i91': -1, 'i92': 1, 'i93': -1, 'i94': 1, 'i95': -1, 'i96': -1, 'i97': -1, 'i98': -1, 'i99': -1, 'i100': -1, 'i101': -1, 'i102': -1, 'i103': -1, 'i104': -1, 'i105': -1, 'i106': -1, 'i107': -1, 'i108': -1, 'i109': -1, 'i110': -1, 'i111': 1, 'i112': -1, 'i113': -1, 'i114': -1, 'i115': -1, 'i116': 1, 'i117': -1, 'i118': -1, 'i119': -1}

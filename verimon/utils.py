def get_pos(labels: list[str]):
    pos = [int(l[5:-1]) for l in labels if len(l) > 5 and l.startswith("[pos")]
    if pos:
        return pos[0]
    else:
        return None

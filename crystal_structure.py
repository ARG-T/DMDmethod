# 結晶構造を選択する
# 結晶構造の名前は https://ja.wikipedia.org/wiki/%E7%B5%90%E6%99%B6%E6%A7%8B%E9%80%A0

def det_structure(inp):
    lat = [(0, 0, 0)]
    # 底心
    if inp[1] == "S":
        lat.append((0, 1/2, 1/2))
    # 体心
    elif inp[1] == "I":
        lat.append((1/2, 1/2, 1/2))
    # 面心
    elif inp[1] == "F":
        lat.append((0, 1/2, 1/2))
        lat.append((1/2, 0, 1/2))
        lat.append((1/2, 1/2, 0))
    # Pは単純
    elif not inp[1] == "P":
        return "Error! Such name structure not exist!"
    return lat

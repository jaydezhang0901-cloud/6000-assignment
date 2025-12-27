# -*- coding: utf-8 -*-
# K-Pop女团歌词提取 - 一代到五代

import os
import json
import csv
from pathlib import Path
from collections import Counter

# 女团列表
GIRL_GROUPS = {
    "一代": [
        "S.E.S.", "S.E.S", "SES", "핑클", "Fin.K.L", "Fin.K.L.",
        "베이비복스", "Baby V.O.X", "Baby VOX", "쥬얼리", "Jewelry",
    ],
    "二代": [
        "소녀시대", "Girls' Generation", "GIRLS' GENERATION", "소녀시대 (GIRLS' GENERATION)",
        "원더걸스", "Wonder Girls", "카라", "KARA", "2NE1",
        "티아라", "T-ara", "T-ARA", "f(x)", "에프엑스",
        "애프터스쿨", "After School", "애프터스쿨 (After School)",
        "포미닛", "4minute", "4Minute", "브라운아이드걸스", "Brown Eyed Girls",
        "시크릿", "Secret", "레인보우", "Rainbow",
        "가비엔제이", "gavy nj", "다비치", "Davichi",
    ],
    "三代": [
        "미쓰에이", "miss A", "Miss A", "씨스타", "SISTAR", "Sistar",
        "걸스데이", "Girl's Day", "Girls Day",
        "에이핑크", "Apink", "APink", "Apink (에이핑크)",
        "EXID", "AOA", "브레이브걸스", "Brave Girls",
        "레이디스 코드", "Ladies' Code", "나인뮤지스", "Nine Muses", "9Muses",
        "크레용팝", "Crayon Pop", "헬로비너스", "Hello Venus",
        "스피카", "Spica", "SPICA", "달샤벳", "Dal Shabet", "피에스타", "FIESTAR",
    ],
    "四代": [
        "레드벨벳", "Red Velvet", "Red Velvet (레드벨벳)",
        "마마무", "MAMAMOO", "Mamamoo", "마마무 (Mamamoo)",
        "트와이스", "TWICE", "TWICE (트와이스)",
        "블랙핑크", "BLACKPINK", "블랙핑크 (BLACKPINK)",
        "여자친구", "GFRIEND", "GFriend", "여자친구 (GFRIEND)",
        "오마이걸", "OH MY GIRL", "Oh My Girl", "오마이걸 (OH MY GIRL)",
        "러블리즈", "Lovelyz", "우주소녀", "WJSN", "Cosmic Girls", "우주소녀 (WJSN)",
        "모모랜드", "MOMOLAND", "Momoland", "모모랜드 (MOMOLAND)",
        "드림캐쳐", "Dreamcatcher", "아이오아이", "I.O.I", "IOI", "아이오아이 (I.O.I)",
        "CLC", "다이아", "DIA", "프리스틴", "PRISTIN", "PRISTIN (프리스틴)",
        "볼빨간사춘기", "Bolbbalgan4", "BOL4", "청하", "Chung Ha", "CHUNGHA",
    ],
    "五代": [
        "아이즈원", "IZ*ONE", "IZONE", "IZ*ONE (아이즈원)",
        "(여자)아이들", "G)I-DLE", "(G)I-DLE", "여자아이들",
        "있지", "ITZY", "ITZY (있지)", "에스파", "aespa", "AESPA",
        "스테이씨", "STAYC", "STAYC(스테이씨)", "아이브", "IVE", "IVE (아이브)",
        "에버글로우", "EVERGLOW", "Everglow",
        "프로미스나인", "fromis_9", "fromis 9", "프로미스나인 (fromis_9)",
        "위클리", "Weeekly", "이달의 소녀", "LOONA", "본월소녀", "이달의 소녀 (LOONA)",
        "로켓펀치", "Rocket Punch", "퍼플키스", "Purple Kiss",
        "빌리", "Billlie", "체리블렛", "Cherry Bullet",
        "시크릿넘버", "Secret Number", "트라이비", "TRI.BE",
        "라잇썸", "Lightsum", "픽시", "Pixy", "PIXY",
        "공원소녀", "GWSN", "네이처", "Nature", "NATURE",
    ],
}

# 建个映射表方便查
artist_map = {}
for gen, names in GIRL_GROUPS.items():
    for name in names:
        artist_map[name.lower()] = gen


def check_group(artist):
    """看看是不是我们要的女团"""
    if not artist:
        return None
    
    a = artist.lower().strip()
    if a in artist_map:
        return artist_map[a]
    
    # 模糊匹配一下
    for gen, names in GIRL_GROUPS.items():
        for name in names:
            if name.lower() in a or a in name.lower():
                return gen
    return None


def get_songs(melon_dir):
    """扫描melon文件夹拿数据"""
    result = []
    cnt = 0
    
    for f in Path(melon_dir).rglob("*.json"):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                d = json.load(fp)
            
            artist = d.get('artist', '')
            gen = check_group(artist)
            
            if gen:
                cnt += 1
                
                # 拿歌词
                lrc = d.get('lyrics', {})
                lines = lrc.get('lines', [])
                txt = '\n'.join(lines) if lines else ''
                
                # 榜单信息
                info = d.get('info', [{}])
                chart = info[0] if info and isinstance(info[0], dict) else {}
                
                result.append({
                    'generation': gen,
                    'artist': artist,
                    'song_name': d.get('song_name', ''),
                    'album': d.get('album', ''),
                    'release_date': d.get('release_date', ''),
                    'genre': d.get('genre', ''),
                    'lyric_writer': d.get('lyric_writer', ''),
                    'composer': d.get('composer', ''),
                    'arranger': d.get('arranger', ''),
                    'year': chart.get('year', ''),
                    'month': chart.get('month', ''),
                    'rank': chart.get('rank', ''),
                    'lyrics': txt,
                    'lyrics_length': len(txt),
                })
                
                if cnt % 50 == 0:
                    print(f"找到 {cnt} 首...")
                    
        except Exception as e:
            print(f"读取出错 {f}: {e}")
    
    print(f"done, 共 {cnt} 首")
    return result


def save_csv(data, path):
    """存成csv"""
    if not data:
        print("没数据!")
        return
    
    cols = ['generation', 'artist', 'song_name', 'album', 'release_date',
            'genre', 'lyric_writer', 'composer', 'arranger',
            'year', 'month', 'rank', 'lyrics', 'lyrics_length']
    
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(data)
    
    size = os.path.getsize(path) / 1024 / 1024
    print(f"保存到 {path}, 大小 {size:.2f}MB")


def show_stats(data):
    """看看统计"""
    print("\n--- 统计 ---")
    
    gen_cnt = Counter(x['generation'] for x in data)
    print("各世代:")
    for g in ["一代", "二代", "三代", "四代", "五代"]:
        print(f"  {g}: {gen_cnt.get(g, 0)}")
    
    artist_cnt = Counter(x['artist'] for x in data)
    print("\nTOP10艺人:")
    for a, c in artist_cnt.most_common(10):
        print(f"  {a}: {c}")


def main():
    print("K-Pop女团数据提取")
    
    # 找melon文件夹
    paths = ["./melon", "./Kpop-lyric-datasets-main/melon", "../melon", "melon"]
    melon = None
    for p in paths:
        if os.path.exists(p):
            melon = p
            break
    
    if not melon:
        print("输入melon路径:")
        melon = input("> ").strip()
    
    if not os.path.exists(melon):
        print(f"找不到 {melon}")
        return
    
    print(f"扫描 {melon} ...")
    
    songs = get_songs(melon)
    
    if songs:
        show_stats(songs)
        save_csv(songs, "girlgroup_songs.csv")
        print("\n完成! 文件: girlgroup_songs.csv")


if __name__ == "__main__":
    main()

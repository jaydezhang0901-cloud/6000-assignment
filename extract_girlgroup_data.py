# -*- coding: utf-8 -*-
"""
K-Pop 女团数据提取脚本
运行此脚本后会生成 girlgroup_songs.csv 文件，上传该文件即可
"""

import os
import json
import csv
from pathlib import Path

# ============================================
# 女团列表（按世代分类）
# ============================================

GIRL_GROUPS = {
    # 一代女团 (1996-2002)
    "一代": [
        "S.E.S.", "S.E.S", "SES",
        "핑클", "Fin.K.L", "Fin.K.L.",
        "베이비복스", "Baby V.O.X", "Baby VOX",
        "쥬얼리", "Jewelry",
    ],
    
    # 二代女团 (2003-2009)
    "二代": [
        "소녀시대", "Girls' Generation", "GIRLS' GENERATION", "소녀시대 (GIRLS' GENERATION)",
        "원더걸스", "Wonder Girls",
        "카라", "KARA",
        "2NE1",
        "티아라", "T-ara", "T-ARA",
        "f(x)", "에프엑스",
        "애프터스쿨", "After School", "애프터스쿨 (After School)",
        "포미닛", "4minute", "4Minute",
        "브라운아이드걸스", "Brown Eyed Girls",
        "시크릿", "Secret",
        "레인보우", "Rainbow",
        "가비엔제이", "gavy nj",
        "다비치", "Davichi",
    ],
    
    # 三代女团 (2010-2013)
    "三代": [
        "미쓰에이", "miss A", "Miss A",
        "씨스타", "SISTAR", "Sistar",
        "걸스데이", "Girl's Day", "Girls Day",
        "에이핑크", "Apink", "APink", "Apink (에이핑크)",
        "EXID",
        "AOA",
        "브레이브걸스", "Brave Girls",
        "레이디스 코드", "Ladies' Code",
        "나인뮤지스", "Nine Muses", "9Muses",
        "크레용팝", "Crayon Pop",
        "헬로비너스", "Hello Venus",
        "스피카", "Spica", "SPICA",
        "달샤벳", "Dal Shabet",
        "피에스타", "FIESTAR",
    ],
    
    # 四代女团 (2014-2017)
    "四代": [
        "레드벨벳", "Red Velvet", "Red Velvet (레드벨벳)",
        "마마무", "MAMAMOO", "Mamamoo", "마마무 (Mamamoo)",
        "트와이스", "TWICE", "TWICE (트와이스)",
        "블랙핑크", "BLACKPINK", "블랙핑크 (BLACKPINK)",
        "여자친구", "GFRIEND", "GFriend", "여자친구 (GFRIEND)",
        "오마이걸", "OH MY GIRL", "Oh My Girl", "오마이걸 (OH MY GIRL)",
        "러블리즈", "Lovelyz",
        "우주소녀", "WJSN", "Cosmic Girls", "우주소녀 (WJSN)",
        "모모랜드", "MOMOLAND", "Momoland", "모모랜드 (MOMOLAND)",
        "드림캐쳐", "Dreamcatcher",
        "아이오아이", "I.O.I", "IOI", "아이오아이 (I.O.I)",
        "CLC",
        "다이아", "DIA",
        "프리스틴", "PRISTIN", "PRISTIN (프리스틴)",
        "우주소녀", "WJSN",
        "볼빨간사춘기", "Bolbbalgan4", "BOL4",
        "청하", "Chung Ha", "CHUNGHA",
    ],
    
    # 五代女团 (2018-2021)
    "五代": [
        "아이즈원", "IZ*ONE", "IZONE", "IZ*ONE (아이즈원)",
        "(여자)아이들", "G)I-DLE", "(G)I-DLE", "여자아이들",
        "있지", "ITZY", "ITZY (있지)",
        "에스파", "aespa", "AESPA",
        "스테이씨", "STAYC", "STAYC(스테이씨)",
        "아이브", "IVE", "IVE (아이브)",
        "에버글로우", "EVERGLOW", "Everglow",
        "프로미스나인", "fromis_9", "fromis 9", "프로미스나인 (fromis_9)",
        "위클리", "Weeekly",
        "본월소녀", "LOONA", "이달의 소녀", "본월소녀 (LOONA)",
        "로켓펀치", "Rocket Punch",
    ],
    
    # 六代女团 (2022-至今)
    "六代": [
        "뉴진스", "NewJeans",
        "르세라핌", "LE SSERAFIM",
        "엔믹스", "NMIXX",
        "케플러", "Kep1er", "Kepler",
        "아일릿", "ILLIT",
        "베이비몬스터", "BABYMONSTER",
        "트리플에스", "tripleS",
        "키스오브라이프", "KISS OF LIFE",
    ],
}

# 创建艺人名到世代的映射
ARTIST_TO_GENERATION = {}
for gen, artists in GIRL_GROUPS.items():
    for artist in artists:
        ARTIST_TO_GENERATION[artist.lower()] = gen

def is_girl_group(artist_name):
    """检查是否为女团"""
    if not artist_name:
        return None
    
    artist_lower = artist_name.lower().strip()
    
    # 直接匹配
    if artist_lower in ARTIST_TO_GENERATION:
        return ARTIST_TO_GENERATION[artist_lower]
    
    # 部分匹配（处理各种变体）
    for gen, artists in GIRL_GROUPS.items():
        for artist in artists:
            if artist.lower() in artist_lower or artist_lower in artist.lower():
                return gen
    
    return None

def extract_songs(melon_path):
    """从melon文件夹提取所有女团歌曲"""
    songs = []
    total_files = 0
    matched_files = 0
    
    melon_path = Path(melon_path)
    
    # 遍历所有JSON文件
    for json_file in melon_path.rglob("*.json"):
        total_files += 1
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            artist = data.get('artist', '')
            generation = is_girl_group(artist)
            
            if generation:
                matched_files += 1
                
                # 提取歌词文本
                lyrics_data = data.get('lyrics', {})
                lyrics_lines = lyrics_data.get('lines', [])
                lyrics_text = '\n'.join(lyrics_lines) if lyrics_lines else ''
                
                # 提取信息
                info = data.get('info', [{}])
                chart_info = info[0] if info and isinstance(info[0], dict) else {}
                
                song = {
                    'generation': generation,
                    'artist': artist,
                    'song_name': data.get('song_name', ''),
                    'album': data.get('album', ''),
                    'release_date': data.get('release_date', ''),
                    'genre': data.get('genre', ''),
                    'lyric_writer': data.get('lyric_writer', ''),
                    'composer': data.get('composer', ''),
                    'arranger': data.get('arranger', ''),
                    'year': chart_info.get('year', ''),
                    'month': chart_info.get('month', ''),
                    'rank': chart_info.get('rank', ''),
                    'lyrics': lyrics_text,
                    'lyrics_length': len(lyrics_text),
                }
                songs.append(song)
                
                # 打印进度
                if matched_files % 50 == 0:
                    print(f"已找到 {matched_files} 首女团歌曲...")
        
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    print(f"\n完成! 共扫描 {total_files} 个文件，找到 {matched_files} 首女团歌曲")
    return songs

def save_to_csv(songs, output_path):
    """保存为CSV文件"""
    if not songs:
        print("没有找到任何女团歌曲!")
        return
    
    fieldnames = [
        'generation', 'artist', 'song_name', 'album', 'release_date',
        'genre', 'lyric_writer', 'composer', 'arranger',
        'year', 'month', 'rank', 'lyrics', 'lyrics_length'
    ]
    
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(songs)
    
    print(f"数据已保存到: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

def print_summary(songs):
    """打印统计摘要"""
    from collections import Counter
    
    print("\n" + "="*50)
    print("数据统计摘要")
    print("="*50)
    
    # 按世代统计
    gen_counts = Counter(s['generation'] for s in songs)
    print("\n各世代歌曲数量:")
    for gen in ["一代", "二代", "三代", "四代", "五代", "六代"]:
        count = gen_counts.get(gen, 0)
        print(f"  {gen}: {count} 首")
    
    # 按艺人统计TOP 10
    artist_counts = Counter(s['artist'] for s in songs)
    print("\n歌曲数量TOP 10艺人:")
    for artist, count in artist_counts.most_common(10):
        print(f"  {artist}: {count} 首")
    
    # 作词家统计TOP 10
    writer_counts = Counter(s['lyric_writer'] for s in songs if s['lyric_writer'])
    print("\n作词家TOP 10:")
    for writer, count in writer_counts.most_common(10):
        print(f"  {writer}: {count} 首")

def main():
    print("="*50)
    print("K-Pop 女团数据提取工具")
    print("="*50)
    
    # ============================================
    # 请修改这里的路径为你的melon文件夹路径
    # ============================================
    
    # 尝试自动找到melon文件夹
    possible_paths = [
        "./melon",
        "./Kpop-lyric-datasets-main/melon",
        "../melon",
        "melon",
    ]
    
    melon_path = None
    for path in possible_paths:
        if os.path.exists(path):
            melon_path = path
            break
    
    if not melon_path:
        print("\n请输入melon文件夹的路径:")
        print("(例如: /Users/yourname/Downloads/Kpop-lyric-datasets-main/melon)")
        melon_path = input("> ").strip()
    
    if not os.path.exists(melon_path):
        print(f"错误: 找不到路径 {melon_path}")
        return
    
    print(f"\n正在扫描: {melon_path}")
    print("这可能需要几分钟，请稍候...\n")
    
    # 提取数据
    songs = extract_songs(melon_path)
    
    if songs:
        # 打印摘要
        print_summary(songs)
        
        # 保存CSV
        output_path = "girlgroup_songs.csv"
        save_to_csv(songs, output_path)
        
        print("\n" + "="*50)
        print("✅ 完成!")
        print(f"请上传 {output_path} 文件到Claude")
        print("="*50)

if __name__ == "__main__":
    main()

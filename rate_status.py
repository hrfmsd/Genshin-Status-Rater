import translations as tr

import aiohttp
import asyncio
import os
import re
import sys
import numpy as np
import math

from cv2 import cv2
from dotenv import load_dotenv
from fuzzywuzzy import fuzz, process
from unidecode import unidecode

load_dotenv()
OCR_API_KEY = os.getenv('OCR_SPACE_API_KEY')

# regex
reg = re.compile(r'\d+(?:[.,]\d+)?')
num_reg = re.compile(r'^(\+)?\d+([.,]\d{1,3})?%?')
bad_reg = re.compile(r'\d+/1000$')
hp_reg = re.compile(r'\d[.,]\d{3}')
atk_reg = re.compile(r'^\+?\d?[.,]?\d{3}')
bad_lvl_reg_1 = re.compile(r'^\+?\d\d?$')
bad_lvl_reg_2 = re.compile(r'^\d{4}\d*$')


# OCR APIに解析してもらった結果を返す
async def ocr(url, num, lang=tr.ja()):
    if not OCR_API_KEY:
        print('Error: OCR_SPACE_API_KEY not found')
        return False, 'Error: OCR_SPACE_API_KEY not found'
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            size = int(r.headers['Content-length'])
            if size > 5e6:
                img = np.asarray(bytearray(await r.read()), dtype="uint8")
                flag = cv2.IMREAD_GRAYSCALE
                if size > 8e6 or os.path.splitext(url)[1] == '.jpg':
                    flag = cv2.IMREAD_REDUCED_GRAYSCALE_2
                img = cv2.imdecode(img, flag)
                _, img = cv2.imencode('.png', img)
                data = aiohttp.FormData()
                data.add_field('apikey', OCR_API_KEY)
                if lang.supported:
                    data.add_field('OCREngine', '2')
                else:
                    data.add_field('language', lang.code)
                data.add_field('file', img.tobytes(), content_type='image/png', filename='image.png')
                ocr_url = f'https://apipro{num}.ocr.space/parse/image'
                async with session.post(ocr_url, data=data) as r:
                    json = await r.json()
            else:
                ocr_url = f'https://apipro{num}.ocr.space/parse/imageurl?apikey={OCR_API_KEY}&url={url}'
                if lang.supported:
                    ocr_url += '&OCREngine=2'
                else:
                    ocr_url += f'&language={lang.code}'
                async with session.get(ocr_url) as r:
                    json = await r.json()
            print(f'OCR Response: {json}')
            if json['OCRExitCode'] != 1:
                return False, f'{lang.err}: ' + '. '.join(json['ErrorMessage'])
            if 'ParsedResults' not in json:
                return False, lang.err_unknown_ocr
            return True, json['ParsedResults'][0]['ParsedText']


# OCR APIの解析結果を解析
def parse(text, lang=tr.ja()):
    results = []
    values = []

    elements = []
    choices = elements + [lang.atk, lang.cr, lang.cd]
    choices = {unidecode(choice).lower(): choice for choice in choices}

    for line in text.splitlines():
        if not line:
            continue

        for k, v in lang.replace.items():
            line = line.replace(k, v)

        line = unidecode(line).lower()
        line = line.replace(':', '.').replace('-', '').replace('0/0', '%')
        line = line.replace('\'', '').replace('*', '')
        if line.replace(' ', '') in lang.ignore or bad_reg.search(line.replace(' ', '')):
            continue

        # 元素熟知の説明文を無視
        has_ignore = False
        for each_reg in lang.ignore_regs:
            each_reg_unidecode = re.compile(re.escape(unidecode(each_reg).lower().replace(' ', '')) + r'(\+\d[.,]\d%)?')
            # print('diff:', each_reg_unidecode, line.replace(' ', ''))
            if each_reg_unidecode.search(line.replace(' ', '')):
                print('ignore:', 'ok')
                has_ignore = True
                break

        if has_ignore:
            continue

        # fuzzywuzzyによる近似探索
        extract = process.extractOne(line, list(choices))
        # print('ext1:', extract, line)
        if extract[1] <= 80:
            extract = process.extractOne(line, list(choices), scorer=fuzz.partial_ratio)
            # print('ext2:', extract)

        if extract[1] > 90:
            stat = choices[extract[0]]
            results += [[stat, '', '']]
            continue

        # 14件（熟知の数値がOCRの問題で取得できない）
        if num_reg.fullmatch(line.replace(' ', '')):
            if line.replace(' ', '') != '0':
                values += [line.replace(' ', '')]

    for result in results:
        # 攻撃力
        if result[0] == lang.atk:
            result[1] = values[1]
            result[2] = values[12].replace('+', '')
            continue

        # 会心率
        if result[0] == lang.cr:
            result[1] = values[4].replace('%', '')
            continue

        # 会心ダメージ
        if result[0] == lang.cd:
            result[1] = values[5].replace('%', '')
            continue

    print(results)
    return results


# スコア付け
def rate(results, options={}, lang=tr.ja()):
    score = 100
    ideal_results = []
    cr = 0.5
    cd = 100
    atk_base = 0
    atk_add = 0

    for result in results:
        stat, value, value_add = result
        # 攻撃力
        if stat == lang.atk:
            atk_base = int(value)
            atk_add = int(value_add)
            continue

        # 会心率
        elif stat == lang.cr:
            cr = float(value) / 100
            continue

        # 会心ダメージ
        if stat == lang.cd:
            cd = float(value) / 100
            continue

    # calc
    atk_rate = 1
    n_atk = atk_base * atk_rate
    x_atk = 0 if atk_base == 0 else (atk_add / atk_base)
    c_atk = n_atk
    exp_dmg_x = calc_exp_dmg(n_atk, x_atk + 0.0466, c_atk, cr, cd)
    exp_dmg_cr = calc_exp_dmg(n_atk, x_atk, c_atk, cr + 0.0311, cd)
    exp_dmg_cd = calc_exp_dmg(n_atk, x_atk, c_atk, cr, cd + 0.0622)
    exp_dmg_v = calc_exp_dmg(n_atk, x_atk, c_atk, cr, cd)

    # 理想値
    a = x_atk + cr * 1.5 + cd * 0.75
    c_n = c_atk / n_atk
    a_ = a + c_n - 1
    if a_ < 2.525:
        ideal_cr = 0.05
        ideal_cd = 0.5
        ideal_x = a_ - 1.5 * 0.05 - 0.375
    elif a_ < 2.776:
        ideal_cr = (a_ - 2.375) / 3
        ideal_cd = 0.5
        ideal_x = a_ - 1.5 * ideal_cr - 0.375
    elif a_ < 4.25:
        ideal_cr = (a_ + 1 + math.sqrt(math.pow((a_ + 1), 2) - 13.5)) / 9
        ideal_cd = ideal_cr * 2
        ideal_x = a - 3 * ideal_cr
    else:
        ideal_cr = 1
        ideal_cd = (a_ - 1.25) * 2 / 3
        ideal_x = a - 0.75 * ideal_cd - 1.5

    ideal_exp_dmg_v = calc_exp_dmg(n_atk, ideal_x, c_atk, ideal_cr, ideal_cd)
    dmg_diff_rate = (exp_dmg_v / ideal_exp_dmg_v - 1)

    print('============')
    print(f'c_n:{c_n}')
    print(f'a_:{a_}')
    print('============')
    # 攻撃力×会心（期待値）：3793
    print(f'exp_dmg_v:{exp_dmg_v}')
    print(f'exp_dmg_x:{exp_dmg_x}')
    print(f'exp_dmg_cr:{exp_dmg_cr}')
    print(f'exp_dmg_cd:{exp_dmg_cd}')
    print(f'ideal_cr:{ideal_cr}')
    print(f'ideal_cd:{ideal_cd}')
    print(f'ideal_exp_dmg_v:{ideal_exp_dmg_v}')
    print(f'dmg_diff_rate:{dmg_diff_rate}')
    print('============')

    # 同じ装備スコアの理想値：+攻撃力(緑)　842
    print(f'+攻撃力(緑)：{int(ideal_x * atk_base):,}')
    # 同じ装備スコアの理想値：会心率　90%
    print(f'会心率：{ideal_cr:.1%}')
    # 同じ装備スコアの理想値：会心ダメージ 180.1%
    print(f'会心ダメージ：{ideal_cd:.1%}')
    # 理想値とのダメージ差：-6.95%
    print(f'理想値とのダメージ差：{dmg_diff_rate:.2%}')

    ideal_results += [int(ideal_x * atk_base)]
    ideal_results += [ideal_cr]
    ideal_results += [ideal_cd]
    ideal_results += [dmg_diff_rate]

    # score
    dmg_diff_rate_score = round(dmg_diff_rate * 100, 1)

    if 0 < dmg_diff_rate_score <= 5:
        score += dmg_diff_rate_score * 2.5
    elif 5 < dmg_diff_rate_score <= 10:
        score += dmg_diff_rate_score * 3
    else:
        score += dmg_diff_rate_score * 5

    print(score, ideal_results)
    return score, ideal_results


# 期待値ダメージ計算
def calc_exp_dmg(n, x, c, cr, cd):
    cr = cr if cr < 1 else 1
    return (n * x + c) * (1 + cr * cd)


if __name__ == '__main__':
    if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    url = 'https://media.discordapp.net/attachments/875974646195970118/882272611646726204/unknown.png'
    # url = 'https://media.discordapp.net/attachments/875974646195970118/882272755163234324/unknown.png'
    lang = tr.ja()
    success, text = asyncio.run(ocr(url, 2, lang))
    # print(text)
    if success:
        results = parse(text, lang)
        rate(results, {}, lang)

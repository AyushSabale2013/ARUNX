#!/usr/bin/env python3

"""
PRADAN Data Download Script

Prerequisites:
* Log in to PRADAN using your web browser.
* Select the required data products (Data order and indexes are meaningful only when same filters and sort order is applied [default are also good]. Indexes may get updated with more data released on PRADAN).
* Download the generated script for the current session.
* Run the script (do not logout the browser session).

Notes:
- Preserves the PRADAN directory structure.
- Resumes interrupted downloads.
- Skips files already downloaded. (Delete the file manually if you want to force re-download or run script in a new directory.)

Caution:
* Session download limits, request rate limits, and session timeouts
  are in effect.
* Excessive automated requests may result in temporary blocking.
* Use this script to simplify manual downloads, but do not overload
  the server.
"""

import requests
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

# =====================================================

# CONFIGURATION

# =====================================================

MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 30
CHUNK_SIZE_MB = 8

url_prefix = "https://pradan1.issdc.gov.in"

# =====================================================

# COOKIES

# =====================================================

cookie_string = "FGTServer=03DE191863F4388C06A7AAAF7E0136FBD15060DF21FA637D82A675307CD5BF28BF8658CAFD950178CA994D;primefaces.download=null;FGTServer=03DE191863F4388C06A7AAAF7E0136FBD15060DF21FA637D82A675307CD5BF28BF8658CAFD950178C9994D;JSESSIONID=f2c40d26728a6627f9186ca5b9fe;JSESSIONID=c4fb247f388448a8435c5bd19a97;OAuth_Token_Request_State=cbbe074f-395a-4531-9ce3-12acaa76bf30;" 

headers = {
"Cookie": cookie_string
}

# =====================================================

# FILE LIST

# =====================================================

data_file_paths = ["/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261031_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/11/N00_0000/AL1_SLX_L1_20261101_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261030_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261029_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261028_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261027_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261026_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260429_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260417_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260220_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260213_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260604_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260423_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261025_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261024_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261023_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261022_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261021_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261020_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261019_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261018_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261017_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261015_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261014_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261013_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261012_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261011_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261010_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261009_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261008_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261007_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261006_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261005_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261004_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261002_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/10/N00_0000/AL1_SLX_L1_20261001_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260930_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260929_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260927_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260926_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260925_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260924_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260921_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260920_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260919_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260918_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260917_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260915_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260914_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260913_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260912_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260911_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260910_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260909_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260908_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260907_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260906_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260905_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260904_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260903_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260902_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/09/N00_0000/AL1_SLX_L1_20260901_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260831_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260830_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260829_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260828_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260827_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260826_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260825_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260823_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260822_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260821_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260820_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260819_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260818_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260817_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260815_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260814_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260813_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260809_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260808_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260807_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260806_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260805_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260804_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260803_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260802_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/08/N00_0000/AL1_SLX_L1_20260801_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260731_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260730_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260729_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260728_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260727_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260726_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260725_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260724_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260723_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260722_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260721_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260720_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260719_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260718_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260717_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260716_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260715_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260714_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260713_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260712_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260711_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260710_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260709_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260708_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260707_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260706_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260705_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260704_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260703_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260702_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/07/N00_0000/AL1_SLX_L1_20260701_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260630_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260627_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260626_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260629_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260628_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260625_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260624_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260623_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260622_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260621_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260620_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260619_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260618_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260617_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260616_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260615_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260614_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260613_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260612_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260611_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260610_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260608_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260607_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260606_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260605_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260603_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260602_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260601_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260531_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260530_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260529_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260528_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260527_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260526_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260525_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260524_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260523_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260521_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260520_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260519_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260518_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260517_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260516_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260515_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260514_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260513_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260512_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260511_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260510_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260509_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260508_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260507_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260506_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260505_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260504_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260503_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260502_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/05/N00_0000/AL1_SLX_L1_20260501_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260430_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260428_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260427_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260426_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260425_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260424_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260422_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260421_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260420_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260419_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260418_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260415_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260414_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260413_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260412_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260411_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260410_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/C26_0090/AL1_SLX_L1_20260409_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/C26_0089/AL1_SLX_L1_20260408_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/C26_0088/AL1_SLX_L1_20260407_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/C26_0081/AL1_SLX_L1_20260406_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260405_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260404_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260403_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260402_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/04/N00_0000/AL1_SLX_L1_20260401_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260331_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260330_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260329_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260328_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260327_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260326_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260325_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260324_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260323_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260322_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260321_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260320_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260319_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260318_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260317_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260316_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260315_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260314_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260313_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260312_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260311_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260310_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260309_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260308_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260307_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260306_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260305_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260304_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260303_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260302_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/03/N00_0000/AL1_SLX_L1_20260301_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260228_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260227_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260226_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260225_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260224_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260223_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260222_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260221_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260219_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260218_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260216_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260215_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260214_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260212_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260211_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260210_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260209_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260208_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260207_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260206_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/C26_0029/AL1_SLX_L1_20260205_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/C26_0025/AL1_SLX_L1_20260203_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260204_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/N00_0000/AL1_SLX_L1_20260202_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/02/C26_0024/AL1_SLX_L1_20260201_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260131_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260130_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260129_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260128_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260127_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260126_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260125_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260124_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260123_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260122_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260121_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260120_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260119_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260118_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260117_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260116_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260115_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260114_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260113_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260112_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260111_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260110_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260109_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260108_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260107_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260106_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260105_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260104_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260103_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260102_v1.0.zip?solexs", "/al1/protected/downloadData/solexs/level1/2026/01/N00_0000/AL1_SLX_L1_20260101_v1.0.zip?solexs", ] 

# =====================================================

# BASE URL

# =====================================================

host_name = urlparse(url_prefix).netloc

# =====================================================

# DOWNLOAD ROOT

# Uses current working directory

# =====================================================

base_dir = Path.cwd()

# =====================================================

# SESSION

# =====================================================

session = requests.Session()

# =====================================================

# STARTUP BANNER

# =====================================================

print("\nPRADAN Bulk Download Utility")
print("-" * 50)
print(f"Download root    : {base_dir / host_name}")
print("Completed files  : skipped (Delete the file manually if you want to force re-download or run script in a new directory.)")
print("Partial files    : resumed")
print("Keep-alive       : enabled")
print("-" * 50)

# =====================================================

# KEEP ALIVE THREAD

# =====================================================


def keep_alive():
    keep_alive_url = (
        url_prefix +
        "/al1/protected/payload.xhtml"
    )

    for _ in range(144):  # 24 hours

        time.sleep(600)  # 10 minutes

        try:
            session.get(
                keep_alive_url,
                headers=headers,
                timeout=(30, 60)
            )

        except Exception as e:
            print(
                f"\n[KEEP-ALIVE ERROR] {e}"
            )

threading.Thread(
target=keep_alive,
daemon=True
).start()

# =====================================================

# DOWNLOAD LOOP

# =====================================================

download_count = 0

for file_index, file_path in enumerate(data_file_paths, start=1):

    url = url_prefix + file_path

    clean_path = file_path.split("?")[0]
    relative_path = clean_path.lstrip("/")

    final_file = (
        base_dir /
        host_name /
        relative_path
    )

    partial_file = Path(
        str(final_file) + ".part"
    )

    final_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n==================================================")
    print(f"File {file_index}/{len(data_file_paths)}")
    print(url)

    # -------------------------------------------------
    # Skip already completed downloads
    # -------------------------------------------------

    if final_file.exists():

        print(
        "Already downloaded. Skipping."
        )

        download_count += 1
        continue

    # -------------------------------------------------
    # Download with retries
    # -------------------------------------------------

    success = False

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            resume_from = 0

            request_headers = headers.copy()

            if partial_file.exists():

                resume_from = (
                    partial_file.stat().st_size
                )

                request_headers[
                    "Range"
                ] = (
                    f"bytes={resume_from}-"
                )

                print(
                    f"Resuming from "
                    f"{resume_from:,} bytes"
                )

            else:

                print(
                    "Starting fresh download"
                )

            with session.get(
                url,
                headers=request_headers,
                stream=True,
                timeout=(30, 600),
                allow_redirects=False
            ) as response:

                if response.status_code not in (
                    200,
                    206
                ):
                    raise RuntimeError(
                        f"HTTP "
                        f"{response.status_code}"
                    )

                mode = (
                    "ab"
                    if resume_from > 0
                    else "wb"
                )

                downloaded = resume_from

                with open(
                    partial_file,
                    mode
                ) as f:

                    for chunk in response.iter_content(
                        chunk_size=
                        CHUNK_SIZE_MB
                        * 1024
                        * 1024
                    ):

                        if not chunk:
                            continue

                        f.write(chunk)

                        downloaded += len(chunk)

                        print(
                            f"\rDownloaded: "
                            f"{downloaded/(1024**2):.3f} MB",
                            end="",
                            flush=True
                        )

            print()

            partial_file.rename(
                final_file
            )

            size_mb = (
                final_file.stat().st_size
                / (1024 ** 2)
            )

            print(
                f"Completed "
                f"({size_mb:.4f} MB)"
            )

            download_count += 1
            success = True

            break

        except Exception as e:

            print()
            print(
                f"Attempt "
                f"{attempt}/{MAX_RETRIES} "
                f"failed: {e}"
            )

            if attempt < MAX_RETRIES:

                print(
                    f"Retrying in "
                    f"{RETRY_WAIT_SECONDS} "
                    f"seconds..."
                )

                time.sleep(
                    RETRY_WAIT_SECONDS
                )

    if not success:

        print()
        print(
        "Error: Network failure, "
        "limits reached, or "
        "session expired."
        )

        print()
        print(
        f"Last file being "
        f"processed "
        f"({file_index}):"
        )

        print(clean_path)

        print()
        print("You may login again later to download script for the new session and resume downloads.")

        raise SystemExit(1)

# =====================================================

# COMPLETION SUMMARY

# =====================================================

print()
print(
f"Your downloads "
f"({download_count}) "
f"are complete."
)

print()
print(
"Downloaded files are "
"available under:"
)

print(base_dir / host_name)

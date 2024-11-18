---
title: Sentinel-2 L1C True Color Optimized
parent: Sentinel-2
grand_parent: Sentinel
layout: script
permalink: /sentinel-2/l1c_optimized/
nav_exclude: true
examples:
- zoom: '12'
  lat: '-2.65054'
  lng: '-42.67467'
  datasetId: S2L1C
  fromTime: '2022-08-07T00:00:00.000Z'
  toTime: '2022-08-07T23:59:59.999Z'
  platform:
  - CDSE
  - EOB
  evalscript: Ly9WRVJTSU9OPTMKZnVuY3Rpb24gc2V0dXAoKSB7CiAgcmV0dXJuIHsKICAgIGlucHV0OiBbIkIwNCIsIkIwMyIsIkIwMiIsICJkYXRhTWFzayJdLAogICAgb3V0cHV0OiB7IGJhbmRzOiA0IH0KICB9Owp9CgovLyBDb250cmFzdCBlbmhhbmNlIC8gaGlnaGxpZ2h0IGNvbXByZXNzCgpjb25zdCBtYXhSID0gMy4wOyAvLyBtYXggcmVmbGVjdGFuY2UKY29uc3QgbWlkUiA9IDAuMTM7CmNvbnN0IHNhdCA9IDEuMzsKY29uc3QgZ2FtbWEgPSAyLjM7CgovLyByZW1vdmUgdGhlIG1pbmltdW0gUmF5bGVpZ2ggc2NhdHRlcmluZyAoY2hlY2sgdGhlIEhpbWFsYXlhcykKY29uc3QgcmF5ID0geyByOiAwLjAxMywgZzogMC4wMjQsIGI6IDAuMDQxIH07CgpmdW5jdGlvbiBldmFsdWF0ZVBpeGVsKHNtcCkgewogIGNvbnN0IHJnYkxpbiA9IHNhdEVuaChzQWRqKHNtcC5CMDQgLSByYXkuciksIHNBZGooc21wLkIwMyAtIHJheS5nKSwgc0FkaihzbXAuQjAyIC0gcmF5LmIpKTsKICByZXR1cm4gW3NSR0IocmdiTGluWzBdKSwgc1JHQihyZ2JMaW5bMV0pLCBzUkdCKHJnYkxpblsyXSksIHNtcC5kYXRhTWFza107Cn0KCmNvbnN0IHNBZGogPSAoYSkgPT4gYWRqR2FtbWEoYWRqKGEsIG1pZFIsIDEsIG1heFIpKTsKCmNvbnN0IGdPZmYgPSAwLjAxOwpjb25zdCBnT2ZmUG93ID0gTWF0aC5wb3coZ09mZiwgZ2FtbWEpOwpjb25zdCBnT2ZmUmFuZ2UgPSBNYXRoLnBvdygxICsgZ09mZiwgZ2FtbWEpIC0gZ09mZlBvdzsKCmNvbnN0IGFkakdhbW1hID0gKGIpID0+IChNYXRoLnBvdygoYiArIGdPZmYpLCBnYW1tYSkgLSBnT2ZmUG93KS9nT2ZmUmFuZ2U7CgovLyBTYXR1cmF0aW9uIGVuaGFuY2VtZW50CmZ1bmN0aW9uIHNhdEVuaChyLCBnLCBiKSB7CiAgY29uc3QgYXZnUyA9IChyICsgZyArIGIpIC8gMy4wICogKDEgLSBzYXQpOwogIHJldHVybiBbY2xpcChhdmdTICsgciAqIHNhdCksIGNsaXAoYXZnUyArIGcgKiBzYXQpLCBjbGlwKGF2Z1MgKyBiICogc2F0KV07Cn0KCmNvbnN0IGNsaXAgPSAocykgPT4gcyA8IDAgPyAwIDogcyA+IDEgPyAxIDogczsKCi8vY29udHJhc3QgZW5oYW5jZW1lbnQgd2l0aCBoaWdobGlnaHQgY29tcHJlc3Npb24KZnVuY3Rpb24gYWRqKGEsIHR4LCB0eSwgbWF4QykgewogIHZhciBhciA9IGNsaXAoYSAvIG1heEMsIDAsIDEpOwogIHJldHVybiBhciAqIChhciAqICh0eC9tYXhDICsgdHkgLTEpIC0gdHkpIC8gKGFyICogKDIgKiB0eC9tYXhDIC0gMSkgLSB0eC9tYXhDKTsKfQoKY29uc3Qgc1JHQiA9IChjKSA9PiBjIDw9IDAuMDAzMTMwOCA/ICgxMi45MiAqIGMpIDogKDEuMDU1ICogTWF0aC5wb3coYywgMC40MTY2NjY2NjY2NikgLSAwLjA1NSk7
---

## Description

True color composite uses visible light bands red, green and blue in the corresponding red, green and blue color channels, resulting in a natural colored product, that is a good representation of the Earth as humans would see it naturally. This visualization uses highlight compression to ensure no maxing-out of clouds or snow, adds the offset to the RGB reflectances to improve the contrast and the color vividness, uses sRGB encoding for no extra darkening of shadows and adds a small amount of saturation boost.

## Description of representative images

Optimized True Color image of Rome. Acquired on 2023-01-30, processed by Sentinel Hub. 

![S2-L1C True Color Optimized](fig/fig1.png)
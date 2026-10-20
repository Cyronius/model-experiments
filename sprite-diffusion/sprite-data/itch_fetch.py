# Download the uploads of a free ("name your price") itch.io asset page, the way the
# "No thanks, just take me to the downloads" button does. Usage:
#   itch_fetch.py <page url> <out dir> [substring filter for upload names]
import re, sys, os, requests
page, out = sys.argv[1].rstrip('/'), sys.argv[2]
filt = sys.argv[3].lower() if len(sys.argv) > 3 else ''
s = requests.Session(); s.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
html = s.get(page).text
csrf = re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)
dl = s.post(page + '/download_url', data={'csrf_token': csrf}).json()['url']
dhtml = s.get(dl).text
csrf = re.search(r'name="csrf_token" value="([^"]+)"', dhtml).group(1)
uploads = re.findall(r'data-upload_id="(\d+)"[^>]*>Download.*?title="([^"]+)"', dhtml, re.S)
print('uploads:', uploads)
os.makedirs(out, exist_ok=True)
for uid, name in uploads:
    if filt and filt not in name.lower(): continue
    # Free packs answer without the download key; passing it gets "invalid key".
    url = s.post(f'{page}/file/{uid}', params={'source': 'game_download', 'as_props': 1,
                 'after_download_lightbox': 'true'}, data={'csrf_token': csrf}).json()['url']
    dst = os.path.join(out, name)
    with s.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dst, 'wb') as f:
            for chunk in r.iter_content(1 << 20): f.write(chunk)
    print('saved', dst, os.path.getsize(dst))

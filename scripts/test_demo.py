import os, sys, re
sys.path.insert(0, '.')
os.environ['DJANGO_SETTINGS_MODULE'] = 'config.settings'
import django; django.setup()
from django.test import Client
c = Client()
r = c.get('/demo/')
print('Status:', r.status_code)
if r.status_code != 200:
    print(r.content.decode('utf-8', 'replace')[:1000])
else:
    content = r.content.decode('utf-8', 'replace')
    imgs = re.findall(r'src="(https://static\.klix\.ba[^"]+)"', content)
    print(f'Slike u HTML-u: {len(imgs)}')
    for url in imgs[:3]:
        print(' ', url)

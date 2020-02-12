## �ڷ������˲���ģ��

ģ���ڷ�������ѵ�������𵽷�������Ҳ������ף�һ��ֻ��Ҫ����һ���ܹ������ƶϵ���С�����������ѵ����������dockerfile�ļ���
Ȼ���޸Ĳ��Դ��룬Լ��ģ�Ͷ��������ʽ��ͨ��web�ӿڵ��ã�ʵ�ֵ��á��ƶϻ������룬ʹ�û�����Ҫӵ��ģ�͵�ѵ���������ʹ��ģ�͡�

CVToolsĿ������С����ѵ���ô�����޸ģ�ʹ�����ײ���CVTools�ڲ�ʹ��Flask Web��ܣ������ṩRestful��ʽ�ӿڡ�
ʹ�������з�ʽ����ģ�ͣ��������£�

```
cvtools -d model.py

```

Ĭ��ʹ��5000�˿ڣ���ʹ��`-p`����ָ����Web��־�Լ���ʱ�ļ�Ĭ���������ǰĿ¼deploy�ļ������£���ʹ��-l����ָ������ȷ�����н�����£�

![img](./_static/2020-02-12_143252.png)

����������£�

```
cvtools -d model.py -p 666 -l deploy/model.log
```

-dָ����ģ��py���룬·����ʹ�����·���������cvtools����ʹ�õ�·����ģ�ʹ�������ṩ`model`���󣬱�����detect�����������Լ̳�
`cvtools.web.model.Model`��

```python
class Model(object):
    """Just as an interface, you have to implement specific model code"""

    def detect(self, img):
        raise NotImplementedError("detect is not implemented!")

    def prase_results(self, results):
        return results

    def draw(self, img, results):
        return img


model = Model()
```

CVTools�ṩ����Web�ӿڣ�

- ip:port: �����������ѡ������ͼƬ������ģ��draw���������
- ip:port/detect: API������ģ�͵�detect���������ؽ����Python���ô���ʾ�����£�

```python
import requests


REST_API_URL = 'http://localhost:666/detect'


image_path = "path/to/image"
# Initialize image path
image = open(image_path, 'rb').read()
form = {'filename': image_path}  # �Ǳ���
multipart = {'image': image}  # �����еĲ���

# Submit the request.
r = requests.post(REST_API_URL, data=form, files=multipart).json()

# Ensure the request was successful.
if r['success']:
    # Loop over the predictions and display them.
    print(r['results'])
# Otherwise, the request failed.
else:
    print('Request failed')
```

- ip:port/show/<string:filename>: �������������������в鿴��������̨�Ѽ���ͼƬ

Note:
- �ӿ����ƺ��������ܻ�仯���ᱣ�ִ�ҳ����¡�
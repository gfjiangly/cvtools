## 文件IO

对pickle、json以及内建的open函数等常用模块和函数的包装，简化代码。

### 读
内建的open包装
- readlines
- read_file_to_list
- read_files_to_list
- read_key_value

json和pickle的包装
- load_json
- load_pkl


### 写
内建的open包装
- write_str
- write_list_to_file
- write_key_value
 
json和pickle的包装
- dump_json
- dump_pkl

例子
```python
import cvtools
import os.path as osp

current_path = osp.dirname(__file__)

# test write_list_to_file
str_list = ['write_list_to_file', 'read_file_to_list']
cvtools.write_list_to_file(str_list, current_path + '/out/io/str_list.txt')

# test read_file_to_list
data = cvtools.read_file_to_list(current_path + '/out/io/str_list.txt')
assert isinstance(data, list)
assert isinstance(data[0], str)
assert len(data) == 2

# test write_key_value
dict_data = {'cat1': 2, 'cat2': 6}
cvtools.write_key_value(dict_data, current_path + '/out/io/dict.txt')

# test read_key_value
data = cvtools.read_key_value(current_path + '/out/io/dict.txt')
assert isinstance(data, dict)
assert isinstance(data['cat1'], str)    # 读出的是字符串
assert len(data) == 2

# 如果上面函数一直成对使用，建议使用序列化读写

# test write_str
str_data = 'str1\nstr2\n'
cvtools.write_str(str_data, current_path + '/out/io/str.txt')
assert osp.isfile(current_path + '/out/io/str.txt')

# test read_files_to_list
# 也可以手动指定要读取的文件，放入list
files = cvtools.get_files_list(current_path + '/out/io')
data_list = cvtools.read_files_to_list(files)
assert len(data_list) == 6

# test readlines
str_data = cvtools.readlines(current_path + '/out/io/str.txt')
assert len(str_data) == 2

# test dump_json
str_data = 'str1\nstr2\n'
cvtools.dump_json(str_data, current_path + '/out/io/str.json')
dict_data = {'cat1': 2, 'cat2': 6}
cvtools.dump_json(dict_data, current_path + '/out/io/dict.json')

# test load_json
str_list = cvtools.load_json(current_path + '/out/io/str.json')
assert isinstance(str_list, str)
dict_data = cvtools.load_json(current_path + '/out/io/dict.json')
assert isinstance(dict_data, dict)
assert isinstance(dict_data['cat1'], int)

# test dump_pkl
str_data = 'str1\nstr2\n'
cvtools.dump_pkl(str_data, current_path + '/out/io/str.pkl')
dict_data = {'cat1': 2, 'cat2': 6}
cvtools.dump_pkl(dict_data, current_path + '/out/io/dict.pkl')

# test load_pkl
str_data = cvtools.load_pkl(current_path + '/out/io/str.pkl')
assert isinstance(str_data, str)
dict_data = cvtools.load_pkl(current_path + '/out/io/dict.pkl')
assert isinstance(dict_data, dict)
assert isinstance(dict_data['cat1'], int)
```

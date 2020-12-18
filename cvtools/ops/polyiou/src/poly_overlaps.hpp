// 五参数表示法，测试见：https://github.com/dingjiansw101/AerialDetection/blob/master/DOTA_devkit/poly_overlaps_test.py
void _overlaps(float* overlaps,const float* boxes,const float* query_boxes, int n, int k, int device_id);
void _overlaps8(float* overlaps, const float* boxes, const float* query_boxes, int n, int k, int device_id);

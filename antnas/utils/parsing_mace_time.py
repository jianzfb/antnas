# @Time    : 2020/3/31 18:52
# @Author  : zhangchenming
import os
import json

file = open('3hs6')
try:
    irb = {}
    irb_skip = {}
    time_sum = 0.0
    input_shape = '[1, 512, 512, 3]'
    i = 0
    for line in file:
        i += 1
        if i <= 5:
            continue
        if i == 6:
            print(line)

        avg_time = line.split('|')[4].strip()
        kernel_size = line.split('|')[10].strip()

        time_sum += float(avg_time)

        if kernel_size != '' and kernel_size[-2] == '7':
            out_shape = line.split('|')[11].strip()

            if input_shape != '[1, 512, 512, 3]':
                time_sum -= float(avg_time)
                name = str(int(eval(input_shape)[-2] / 512 * 224)) + 'x' + str(eval(input_shape)[-1]) + 'x' + str(
                    int(eval(out_shape)[-2] / 512 * 224)) + 'x' + str(eval(out_shape)[-1])

                if eval(input_shape)[-2] == eval(out_shape)[-2] and eval(input_shape)[-1] == eval(out_shape)[-1]:
                    irb_skip[name] = float('%.3f' % time_sum)
                else:
                    irb[name] = float('%.3f' % time_sum)

            time_sum = 0
            input_shape = out_shape

    json.dumps({'latency': irb})
    irb_result = {'latency': irb_skip}
    json.dumps({'': irb_result})

except IndexError as e:
    print(e)



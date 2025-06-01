[toc]

# 概述

# ezsholarsearch文件夹



## scholarly与search

由于谷歌现在几乎完全禁止自动程序访问google scholar，scholarly的绝大部分功能已经处于不可用状态；即使使用cookies进行模拟登陆，操作也会受到严重的限制。好在在保证有梯子的情况下，我们的search_pub是一般可用的，advanced是少数时候可用的，其他功能则有极高概率被谷歌拦截并拒绝访问。

search_test（位于主文件夹EZScholarSearch-main下）：针对search的测试程序，为了避免被谷歌拦截，测试程序只包装了search_pub这个函数。search_pub本质上是进行一般搜索，所以不使用文章名也是可以的，甚至可以搜索作者名（相对的，直接author_search则会被谷歌精准拦截）

一个预期的测试输出：

INFO:scholarly:Getting https://scholar.google.com/scholar?hl=en&q=yunfeng%20xiao&as_vis=0&as_sdt=0,33
INFO:httpx:HTTP Request: GET https://scholar.google.com/scholar?hl=en&q=yunfeng%20xiao&as_vis=0&as_sdt=0,33 "HTTP/1.1 200 OK"
INFO:scholarly:Getting https://scholar.google.com/scholar?hl=en&q=info:MWrYX4uuVuMJ:scholar.google.com/&output=cite&scirp=0&hl=en

.......

只要有cookies，它也会使用cookies并显示，无论cookies是否有效。

search虽然完成了较多功能，但遗憾的是在google的封锁下我们难以对它进行充分且稳定的测试。

# cookies tool

这个文件夹是帮助生成本人google scholar的cookies的。需要注意的是，只有google scholar处于登录状态下（不是右上角，而是能够点进我的个人学术档案并访问内容），浏览器保存的cookies才是有效的。

## transform_json

将cookies转换为json文件方便读取和操作，这一文件适配的是windows chorme，注意更改至自己的路径。

## cookies_check

检查cookies是否有效，是否包含了google scholar的登录信息。
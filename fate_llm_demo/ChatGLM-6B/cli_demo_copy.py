import os
import platform
import signal
from transformers import AutoTokenizer, AutoModel
import os
path = os.path.dirname(__file__)

tokenizer = AutoTokenizer.from_pretrained(os.path.join(path, "chatglm-6b"), trust_remote_code=True)
model = AutoModel.from_pretrained(os.path.join(path, "chatglm-6b"), trust_remote_code=True).half().quantize(4).cuda()
model = model.eval()

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True

str1 = """
1. 漂亮星辰卡-农业银行
   权益：新户核卡后消费3笔满18元交易，可领取200元线上购物刷卡金（可用于淘宝、京东、美团、唯品会、cdf海南免税）；3月31号前核卡，4月30号前激活的新户，可额外再领取一张50元消费返现；
        新户每月银联渠道消费满666元，可再领取50元线上购物刷卡金（可用于淘宝、京东、美团、唯品会、cdf海南免税）。
   年费：首年免年费，消费8笔免次年。
2. 环球商旅卡-农业银行
   权益：主要用来参加VISA双标卡返现，一般每个月可返现60元；
   年费：首年免年费，消费5笔免次年。
3. 全球支付白 - 建设银行
   权益：全球免收外汇兑换手续费；高额延误险，每延误4小时赔付200元，行李每延误4小时赔付100元。
   年费： 主卡580/年，任意消费、取现10笔免当年年费。
4. bilibili联名卡 - 建设银行
   权益：新户消费6笔银联通道消费可领取1年的B站大会员；银联通道消费满2万元，可额外再领取1年的B站大会员；可参加银联云闪付无界卡活动。
   年费：主卡580/年，任意消费、取现10笔免当年年费。
对于刚刚毕业踏入职场的年轻人或是自由职业者来说， 漂亮星辰卡有三个卡面，都是白金卡级别，新户礼一共可领300元刷卡金，这是实打实的钱，推荐使用漂亮星辰卡。以上仅供参考，当然也可以根据自己的需求选择其他适合自己的信用卡。
"""

str2 = """
投资理财产品是多种多样的，包括债券、银行定期存款、信用卡、股票投资、保险理财等。通常而言，投资理财总是有一定风险的，适合您的投资理财才是最好的。
建议还是应该根据具体产品的特定，依据自身的需求情况和经济能力考虑。
1、明确理财目标，找出理财缺口合理制定目标，明确和目标的差距，同时兼顾短期及长期目标。
2、自测风险承受度，合理规划资产配置组合根据自己的风险承受能力确定资产配置组合，风险依次由小到大为：存款，银行委托理财产品、债券、基金，股票。
3、选择合适理财产品和投资方法根据资产配置组合确定投资品种及配置比例。

若一个人每月可存款2万元，一年24万，加上年终奖最多28w左右，其理财方案如下:
1、每年定期存款20万，每月存1万6。存5年的定期，零存整取那种建议放股份银行，年利率3.3 %左右，大约5年后本息得109万左右。
2、一年留8000千为活期，也是每月6百多，以备不时之需。这笔要能剩下话 5年有个2~3万元
3、 一年用1万元用来投资理财，每月800多元做定投，回报高点话定投基金股票，保守点定投黄金证券，推荐后者。等3~4年后投资类资金超过5w了，就取出来买保本理财。这么算了5年大概保守7w甚至更多吧这么算5年120w元也够付首付了。这么算了等于用钱养钱，把你这5年的日常开销挣回来了。
"""

from prettytable import PrettyTable

# 定义表格的数据
data = [
    ["产品", "权益", "年费", "推荐指数"],
    ['漂亮星辰卡', '新户消费3笔满18元交易,新户消费3笔满18元交易，有三个白金卡级别卡面,新户可领300元刷卡金.', '首年免年费，消费8笔免次年', "*****"],
    ["环球商旅卡", "主要用来参加VISA双标卡返现，一般每个月可返现60元", "首年免年费，消费5笔免次年", "****"],
    ["全球支付白", "全球免收外汇兑换手续费；高额延误险，每延误4小时赔付200元。", "主卡580/年，取现10笔免当年年费。", "***"],
    ["bilibili联名卡", "新户消费6笔可领1年的B站大会员；可参加银联云闪付无界卡活动。", "主卡580/年，取现10笔免当年年费。", "***"],
    ["超惠真金白金卡", "12306购票享9折优惠最多减30，每天限1次;账单分期6折手续费", "消费满5笔或5000元免当年;白金等级有520元制卡费", "***"],
    ["天猫超市卡", "12月31号前核卡的新户，可享3次天猫超市满200元减50元;可参加银联无界卡活动", "首年免年费，12笔刷免", "*****"],
    ["长城无界运动", "截至3月31日，消费达标可领会员季卡，每人限领2次;可参加银联无界卡活动", "首年免年费，12笔刷免", "****"]
]

# 创建表格对象
table1 = PrettyTable()
table1.field_names = data[0]
for row in data[1:]:
    table1.add_row(row)

# 打印表格
print(table1)

summary1 = "对于刚刚毕业踏入职场的年轻人来说， 漂亮星辰卡有三个卡面，都是白金卡级别，" \
           "新户礼一共可领300元刷卡金，这是实打实的钱，推荐使用漂亮星辰卡。以上仅供参考，当然也可以" \
           "根据自己的需求选择其他适合自己的信用卡。"

q1 = "请推荐一款适合应届大学生的信用卡产品"

q2 = "我选择漂亮星辰卡,该怎么申请办卡？"

q3 = "确认执行"

task1 = "您好！办理漂亮星辰卡的流程如下：\n" \
        "1. 个人身份认证：进行人脸识别认证； \n" \
        "2. 信息录入：相关输入个人姓名，年龄，职业等信息；\n" \
        "3. 确认提交申请。\n"  \
        "请问是否需要执行以上操作？"
task2 = "请您打开摄像头，对准摄像头录入人脸"

data1 = [
    ["产品", "描述"],
    ["基金", "债券型基金不属于保本类型的理财产品，虽然波动不会像权益类基金那么大，但因为市场行情的不好影响造成本金短瞬亏损，也是比较常见的"],
    ["智能存款", "智能储蓄存款，一年期的存款利率最高可达5%，比定期理财和大额存单的利率都要高。风险系数低，起投门槛最低只要50元即可，还保本保息"],
    ["P2P类理财", "封闭期最少3个月起，最长期限为365天，历史年化收益率5.6%-8.5%不等;如果平台安全可靠的话，那么可以进行投资的"],
    ["银行定存", "银行存款是最基本的理财方式。每个月存入一笔资金进去,从第二年的第一个月开始，每月都能获得可观收益。既保证了收益，也保证了流动性"]
]

# 创建表格对象
table2 = PrettyTable()
table2.field_names = data1[0]
for row in data1[1:]:
    table2.add_row(row)

summary2 = """
1、每年银行定存20万，每月存1万6。存5年的定期，零存整取那种建议放股份银行，年利率3.3 %左右，大约5年后本息得109万左右。
2、一年留8000千为活期，也是每月6百多，以备不时之需。这笔要能剩下话 5年有个2~3万元
3、一年用1万元用来投资理财，每月800多元做定投，回报高点话定投基金股票，保守点定投黄金证券，推荐后者。等3~4年后投资类资金超过5w了，就取出来买保本理财。这么算了等于用钱养钱。
"""

def main():
    history = []
    global stop_stream
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        if "信用卡" in query.strip():
            print("\nFedAssistant: ")
            print(table1)
            print("推荐建议: \n")
            print(summary1)
        if "申请办卡" in query.strip():
            print("\nFedAssistant: ")
            print(task1)
        if "执行" in query.strip():
            print("\nFedAssistant: ")
            print(task2)
        if "投资理财" in query.strip() and "月存款" not in query.strip():
            print("\nFedAssistant: ")
            print(table2)
        if "月存款" in query.strip():
            print("\nFedAssistant: ")
            print(summary2)
        count = 0
        for response, history in model.stream_chat(tokenizer, query, history=history):
            if stop_stream:
                stop_stream = False
                break
            #if "信用卡" in query.strip():
            #    print(str1)
            else:
                count += 1
                if count % 8 == 0:
                    #os.system(clear_command)
                    #print(build_prompt(history), flush=True)
                    signal.signal(signal.SIGINT, signal_handler)
        #os.system(clear_command)
        #print(build_prompt(history), flush=True)


if __name__ == "__main__":
    main()

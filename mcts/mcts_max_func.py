import pylab as pl  # 导入 pylab 库,用于绘图
from treelib import Tree  # 导入 Tree 类,用于创建树结构
import numpy as np  # 导入 numpy 库,用于数值计算

"""
求函数的最大值
"""

class Data(object):
    # 定义 Data 类,用于存储节点的数据
    def __init__(self, domain=0, visits=0, best_score=(-np.inf), coef=2, is_terminal=False):
        # 初始化 Data 对象
        self.visits = visits  # 访问次数
        self.best_score = best_score  # 最佳得分
        self.coef = coef  # UCB1 公式中的系数
        self.domain = domain  # 节点的域

    def ucb(self, parent_visits):
        # 计算 UCB1 分数
        if self.visits == 0:
            return np.inf  # 如果未访问过,返回无穷大
        return self.best_score + self.coef * np.sqrt(np.log(parent_visits)/self.visits)    

class MCTS(object):
    # 定义 MCTS 类,实现蒙特卡洛树搜索算法
    def __init__(self, func, domain, max_depth=10, rollout_times=10):
        # 初始化 MCTS 对象
        self.max_depth = max_depth  # 最大深度
        self.tree = Tree()  # 创建树结构
        self.func = func  # 目标函数
        self.domain = domain  # 搜索域
        self.rollout_times = rollout_times  # 每次模拟的次数
        self.root = self.tree.create_node('root', data=Data(domain=domain))  # 创建根节点

    def train(self, steps=100):
        # 训练 MCTS
        for n in range(steps):
            node = self.root 
            while not self.is_terminal(node):
                node = self.traverse(node)
                score = self.rollout(node)
                self.back_propagate(node, score)

    def get_optimal(self):
        # 获取最优解
        node = self.traverse(self.root, greedy=True)
        return np.mean(node.data.domain), node.data.best_score
        
    def expand(self, node):
        # 扩展节点
        domain_left = [node.data.domain[0], (node.data.domain[0]+node.data.domain[1])/2]
        domain_right = [(node.data.domain[0]+node.data.domain[1])/2, node.data.domain[1]]
        left_node = self.tree.create_node('left', parent=node, data=Data(domain=domain_left))
        right_node = self.tree.create_node('right', parent=node, data=Data(domain=domain_right))
        return left_node
            
    def traverse(self, node, greedy=False):
        # 遍历树
        while True:
            if self.is_terminal(node):
                return node
            if not self.is_fully_expanded(node):
                return self.expand(node)
            node = self.get_best_child(node, greedy=greedy)

    def is_fully_expanded(self, node):
        # 检查节点是否完全扩展
        return bool(self.tree.children(node.identifier))

    def is_terminal(self, node):
        # 检查节点是否为终端节点
        return self.tree.level(node.identifier) == self.max_depth

    def back_propagate(self, node, score):
        # 反向传播更新节点信息
        while True:
            node.data.best_score = max(node.data.best_score, score)
            node.data.visits += 1
            if node.is_root():
                break
            node = self.tree.parent(node.identifier)
        
    def get_best_child(self, node, greedy):
        # 获取最佳子节点
        best_child = None
        children = self.tree.children(node.identifier)
        if children:
            parent_visits = node.data.visits
            if greedy:
                scores = [child.data.best_score for child in children]
            else:
                scores = [child.data.ucb(parent_visits) for child in children]
            best_child = children[np.argmax(scores)]
        return best_child
            
    def rollout(self, node):
        # 执行随机模拟
        domain = node.data.domain
        scores = []
        for n in range(self.rollout_times):
            x = domain[0] + np.random.random()*(domain[1]-domain[0])
            score = self.func(x)
            scores.append(score)
        return np.max(scores)
            
        
def func(x):
    # 定义目标函数
    return -np.sin(2*x*np.pi) - x


mcts = MCTS(func, [-1, 1], max_depth=16)  # 创建 MCTS 实例
mcts.train()  # 训练 MCTS
x_best, y = mcts.get_optimal()  # 获取最优解

print('The optimal solution is ~ {:.5f}, which is located at x ~ {:.5f}.'.format(y, x_best))  # 打印最优解

x = np.linspace(-1, 1, 100)  # 生成 x 轴数据
_ = pl.plot(x, func(x))  # 绘制函数图像
_ = pl.scatter(x_best, y, c='r')  # 标记最优点
_ = pl.grid()  # 添加网格
_ = pl.xlabel('x')  # 设置 x 轴标签
_ = pl.ylabel('y')  # 设置 y 轴标签
pl.savefig('fig.png')  # 保存图像

class Car():
    '''汽车类'''
    def __init__(self,rank = '小型车',color = 'white',brand = '比亚迪'):
        self.rank = rank  # 车型
        self.color = color # 颜色
        self.brand =  brand # 品牌
        self.mileage = 10  # 行驶里程（单位：千米）
    def get_mileage(self):
        print('行驶里程为',self.mileage,'千米。')
        
    def set_mileage(self,value):
        self.mileage += value
        
if __name__ == '__main__':
    print('='*5,'第一个实例','='*5)
    car = Car()
    print('车型：',car.rank)  # 输出属性
    print('颜色：',car.color)  # 输出属性
    print('品牌：',car.brand)  # 输出属性

    # 调用方法
    car.get_mileage()     # 输出原行驶里程
    car.set_mileage(5.6)  # 行驶里程增加5.6千米
    print('更改行驶里程……')
    car.get_mileage()     # 输出新行驶里程

    print('='*5,'第二个实例','='*5)
    car = Car('SUV','黑色','丰田')
    print('车型：',car.rank)  # 输出属性
    print('颜色：',car.color)  # 输出属性
    print('品牌：',car.brand)  # 输出属性

    # 调用方法
    car.get_mileage()     # 输出原行驶里程
    car.set_mileage(10)  # 行驶里程增加10千米
    print('更改行驶里程……')
    car.get_mileage()     # 输出新行驶里程

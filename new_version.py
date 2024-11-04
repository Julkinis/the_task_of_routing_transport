import datetime
import collections
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
import xlrd
from matplotlib.font_manager import FontProperties

class Node:
    """  Класс для хранения узла клиента """ 
    def __init__(self, id, coordinates, volume):
        # id клиента
        self.id = id
        # адрес 
        self.coordinates = coordinates
        # дорога из депо к клиенту
        self.dn_edge = None
        # дорога от клиента к депо  
        self.nd_edge = None
        # маршрут, в который входит узел
        self.route = None
        # объем товара
        self.volume = volume

    def __repr__(self):
        return str(self.id)

    
class Edge:
    """  Класс для хранения ребра """ 
    def __init__(self, origin, destination, cost, saving = 0):
        # узел начала
        self.origin = origin
        # узел конца 
        self.destination = destination
        # значение выигрыша, связанное с этим ребром
        self.saving = saving
        # длина ребра
        self.cost = cost
        # обратное ребро
        self.inv = None
        
    def inverse(self):
        self.origin, self.destination = self.destination, self.origin

    def __repr__(self):
        return f"({self.origin} -> {self.destination})"
    

class Route:
    # хранит ребра и стоимость маршрута
    def __init__(self, edges = None, depot = None):
        self.edges = edges or collections.deque()
        self.cost = sum(edge.cost for edge in self.edges)
        # объем товаров 
        self.volume = sum(edge.destination.volume for edge in self.edges)
        self.depot = depot

    # первый узел, посещенный после склада 
    @property
    def first_node(self):
        return self.edges[0].destination

    # последний узел, посещенный перед возвратом на склад 
    @property
    def last_node(self):
        return  self.edges[-1].origin

    # удалить первое ребро, перерассчитать стоимость 
    def delete_left_edge(self):
        removed = self.edges.popleft()
        self.cost -= removed.cost

    # удалить последнее ребро, перерассчитать стоимость 
    def delete_right_edge(self):
        removed = self.edges.pop()
        self.cost -= removed.cost

    def extend(self, edges):
        # добавить новые ребра, # перерассчитать стоимость 
        self.edges.extend(edges)
        #self.cost += sum(edge.cost for edge in edges)
        
    def inverse(self):
        self.edges.reverse()
        for i in range(len(self.edges)):
            edge = self.edges[i]
            if edge.origin == self.depot or edge.destination == self.depot:
                self.edges[i] = edge.inv
            else:
                self.edges[i].inverse()        

    def merge(self, route):
        # (0->1->3->0)  (0->1->2->0)  (0->3->1->2->0)
        if self.first_node == route.first_node:
            vol = self.first_node.volume
            self.inverse()
            # удалить ребро (0->1) и вычесть стоимость
            self.delete_right_edge()
            # удалить ребро (0->1) и вычесть стоимость
            route.delete_left_edge()
            self.edges.extend(route.edges)
            # пересчитать стоимость
            self.cost += route.cost
            # пересчитать объем товара
            self.volume = self.volume + route.volume - vol
        elif self.last_node == route.first_node:
            vol = self.last_node.volume
            self.delete_right_edge()
            route.delete_left_edge()
            self.edges.extend(route.edges)
            self.cost += route.cost
            self.volume = self.volume + route.volume - vol
        elif self.first_node == route.last_node:
            vol = self.first_node.volume
            self.delete_left_edge()
            route.delete_right_edge()
            route.edges.extend(self.edges)
            route.cost += self.cost
            self.volume = self.volume + route.volume - vol
        elif self.last_node == route.last_node:
            vol = self.last_node.volume
            route.inverse()
            self.delete_right_edge()
            route.delete_left_edge()
            self.edges.extend(route.edges)
            self.cost += route.cost
            self.volume = self.volume + route.volume - vol
        #self.volume = sum(edge.destination.volume for edge in self.edges)


    def append(self, edge):
        # добавить новое ребро, перерассчитать стоимость
        # (2->3) (0->3->0)  (0->2->3->0)
        if edge.destination == self.first_node:
            self.delete_left_edge()
            self.edges.appendleft(edge)
            self.edges.appendleft(edge.origin.dn_edge)
            self.cost += edge.origin.dn_edge.cost
            self.volume += edge.origin.volume
        
        elif edge.origin == self.last_node:
            self.delete_right_edge()
            self.edges.append(edge)
            self.edges.append(edge.destination.nd_edge)
            self.cost += edge.destination.nd_edge.cost
            self.volume += edge.destination.volume
        # (2->3) (0->2->4->0)  (0->3->2->4->0) )
        elif edge.origin == self.first_node:
            self.delete_left_edge()
            edge.inverse()
            self.edges.appendleft(edge)
            self.edges.appendleft(edge.origin.dn_edge)
            self.cost += edge.origin.dn_edge.cost
            self.volume += edge.origin.volume

        elif edge.destination == self.last_node:
            self.delete_right_edge()
            edge.inverse()
            self.edges.append(edge)
            self.edges.append(edge.destination.nd_edge)
            self.cost += edge.destination.nd_edge.cost
            self.volume += edge.destination.volume

        self.cost += edge.cost
        #self.volume = sum(edge.destination.volume for edge in self.edges)      


    def drop(self):
        self.edges = collections.deque([Edge(self.depot, self.depot, 0)])
        self.cost = 0
        self.volume = 0

        
    def __repr__(self):
        return str(list(self.edges))

    
def distance(city1, city2):
    return math.sqrt(math.pow(city1[0] - city2[0], 2) + math.pow(city1[1] - city2[1], 2))


def get_data(path):
        """
        Используйте модуль xlrd для считывания данных из определенного столбца в Excel и возврата
        """
        data = xlrd.open_workbook(path)  # Откройте выбранную таблицу Excel и назначьте ее data
        table = data.sheet_by_index(0)  # Лист 0
        return table.col_values(0), table.col_values(1), table.col_values(2)


def CW_alg(node_x, node_y, node_w, N, Q):
    depot = Node(0, [node_x[0], node_y[0]], 0)
    q = np.array(node_w)
    # print(node_x, node_y, node_w)
    nodes = np.array([])
    edges = np.array([])
    costs = np.array([])
    costs = [[0] * len(node_x) for i in range(len(node_x))]  # Определите двумерный список для хранения расстояния между узлами
    
    for i in range(len(node_x)):
        for j in range(len(node_x)):
            costs[i][j] = distance([node_x[i], node_y[i]], [node_x[j], node_y[j]]) 
    #print(costs)
    # создание узлов и ребер к/от депо
    for i in range(N):
        nodes = np.append(nodes,(Node(i+1, (node_x[i+1], node_y[i+1]), q[i+1])))
        cost = costs[0][nodes[i].id]
        edge_dn = Edge(depot, nodes[i], cost, 0)
        edge_nd = Edge(nodes[i], depot, cost, 0)
        edge_dn.inv = edge_nd
        edge_nd.inv = edge_dn
        edges = np.append(edges, edge_dn)
        edges = np.append(edges, edge_nd)
        nodes[i].dn_edge, nodes[i].nd_edge = edge_dn, edge_nd


    # создание ребер 
    for i, j in itertools.combinations(nodes, 2):
        cost = costs[i.id][j.id]
        saving = i.nd_edge.cost + j.dn_edge.cost - cost
        edge = Edge(i, j, cost, saving)
        edges = np.append(edges, edge)
        

    # сортировка по убыванию выигрышей 
    edges = sorted(edges, key=lambda edge : edge.saving, reverse=True)
    # for i in range(len(nodes)):
    #     print(nodes[i], nodes[i].dn_edge, nodes[i].nd_edge)
    # for i in range(len(edges)):
    #     print(edges[i])
    

    # Генерирует фиктивное решение
    routes = np.array([])
    for node in nodes:
        r = Route(collections.deque([node.dn_edge, node.nd_edge]), depot)
        node.route = r
        routes = np.append(routes, r)


    for i in range(len(edges)):
        flag = False

        # не заблокировано 
        if edges[i].origin != depot and edges[i].destination != depot :    

            # не входят в состав одного и того же маршрута 
            if edges[i].origin.route  != edges[i].destination.route :
                # print(edges[i].origin.route.first_node, edges[i].origin.route.last_node)
                # print(edges[i].destination.route.first_node, edges[i].destination.route.last_node)

                # являются начальной/конечной мрашрутов в которые входят 
                if {edges[i].origin} & {edges[i].origin.route.first_node, edges[i].origin.route.last_node}  \
                    and {edges[i].destination} & {edges[i].destination.route.first_node, edges[i].destination.route.last_node}:

                    # присоединение узла, не вхоящего ни в один составной маршрут
                    if len(edges[i].destination.route.edges) == 2:
                        if edges[i].origin.route.volume + edges[i].destination.volume <= Q:
                            edges[i].destination.route.drop()
                            edges[i].origin.route.append(edges[i])
                            flag = True

                    # присоединение узла, не вхоящего ни в один составной маршрут
                    elif len(edges[i].origin.route.edges) == 2:
                        if edges[i].destination.route.volume + edges[i].origin.volume <= Q:
                            edges[i].origin.route.drop()
                            edges[i].destination.route.append(edges[i])
                            flag = True

                    # объединение маршрутов
                    else:
                        if edges[i].destination.route.volume + edges[i].origin.route.volume <= Q:
                            edges[i].destination.route.append(edges[i])
                            edges[i].origin.route.merge(edges[i].destination.route)
                            edges[i].destination.route.drop()
                            flag = True

                    # запись нового маршрута в ребро
                    if flag == True:
                        if len(edges[i].origin.route.edges) > len(edges[i].destination.route.edges):
                            edges[i].destination.route = edges[i].origin.route
                        else:
                            edges[i].origin.route = edges[i].destination.route
   
    return routes, costs


def heat(old_path, node_w, m, costs, Q):
    """
    Функция нагрева
    :параметр costs: расстояние между узлами
    :параметр m: количество транспортных средств
    :параметр node_w: (изначально определенный) вес каждой точки
    :параметр old_path: Исходное решение, то есть любое решение, заданное при инициализации
    :return: Возвращает начальную температуру T0 и старый путь old_path
    """
    dc = np.zeros(3000)  # Установите количество раз нагрева. В этом примере алгоритм 2-opt используется для нагрева исходного раствора 4000 раз (случайным образом прерывается).

    for i in range(3000):
        new_path = new_paths(old_path)  # Сгенерируйте новый путь
        dis1 = total_dis(old_path, node_w, m, costs, Q)  # Вычислите расстояние по старому пути
        dis2 = total_dis(new_path, node_w, m, costs, Q)  # Вычислите расстояние по новому пути
        dc[i] = abs(dis2 - dis1)  # Отклонение расстояния между старым и новым путями
        old_path = new_path

    T0 = 20 * max(dc)  # Установите начальную температуру в 20 раз превышающую максимальное отклонение

    return T0, old_path


def new_paths(old_path):
    """
    Для генерации нового пути используется алгоритм 2-opt
    : param old_path: старый путь
    : return: Новый путь
    """
    N = len(old_path)
    # Генерируйте случайные целые числа в диапазоне от 2 до N (все пункты плюс максимальное число автомобилей), гарантируя, что начало и конец равны 0
    a, b = np.random.randint(1, N), np.random.randint(1, N)  
    random_left, random_right = min(a, b), max(a, b)  # Отсортируйте сгенерированные целые числа
    rever = old_path[random_left:random_right]  # Путь к средней части случайно выбранного old_path
    new_path = old_path[:random_left] + rever[::-1] + old_path[random_right:]  # 2-optАлгоритм, переверните и проложите новый путь

    return new_path


def total_dis(path, node_w, m, costs, Q):
    """
      Чтобы вычислить функцию расстояния, здесь есть небольшая хитрость, которая заключается 
    в предварительном вычислении расстояния для формирования двумерного списка. 
    В дальнейшем вам нужно будет только найти список для вычисления расстояния, что значительно экономит время.
    :параметр costs: расстояние между узлами
    :параметр m: количество транспортных средств
    :param path: путь, который будет вычислен
    :параметр node_w: (изначально определенный) вес каждой точки
    :return: Целевая функция, то есть значение расстояния текущего пути + синтез элемента штрафа
    """
    H = 0.1 * Q
    dis = 0
    for i in range(len(path) - 1):  # Вычислите расстояние между двумя точками на траектории пути и верните значение расстояния, просмотрев список
        dis += costs[path[i]][path[i + 1]]

    address_index = [i for i in range(len(path)) if path[i] == 0]  # Найдите координаты 0 (склад) в пути
    C = [0] * m  # Вместимость каждого автомобиля
    M = [0] * m  # Установите штрафные санкции, и штрафные санкции будут наложены, если максимальная вместимость каждого автомобиля будет превышена на 200
    #print(len(node_w), len(path), len(address_index))
    for i in range(len(address_index) - 1):  # Число между координатами склада (0) - это маршрут следования каждого автомобиля
        for j in range(address_index[i], address_index[i + 1], 1):
            #print(path[j])
            C[i] += node_w[path[j]]  # Рассчитайте текущую вместимость каждого автомобиля, чтобы гарантировать, что максимальная вместимость в 200 человек не может быть превышена
        if C[i] >= Q:
            M[i] = H * (C[i] - Q)  # H - Штрафная единица для предотвращения превышения максимальной вместимости транспортного средства, равной 200, коэффициент штрафной единицы, равный 20, может быть изменен

    sum_M = sum(M)  # Сумма штрафных санкций

    return dis + sum_M


def metropolis(old_path, new_path, node_w, m, T, costs, Q):
    """
    Критерием метрополиса является вероятность принятия нового решения в моделируемом алгоритме отжига. Чтобы предотвратить попадание в локальное оптимальное решение, необходимо принять новое решение с определенной вероятностью.
    :параметр old_path: старый путь
    :параметр new_path: новый путь
    :параметр node_w: Спрос на товары в магазине
    :параметр m: Количество транспортных средств
    :параметр T: Текущая температура при моделируемом внешнем цикле отжига
    :параметр costs: расстояние между узлами
    :return: Возвращает текущее оптимальное решение и соответствующую целевую функцию (расстояние), оцениваемую по критерию метрополии
    """
    dis1 = total_dis(old_path, node_w, m, costs, Q)  # Расстояние по старому пути
    dis2 = total_dis(new_path, node_w, m, costs, Q)  # Расстояние нового пути

    dc = dis2 - dis1  # Разница между этими двумя

    if dc < 0 or np.exp(-abs(dc) / T) > np.random.random():  # Рекомендации Metropolis, две ситуации, в которых принимается новое решение: 1.Расстояние нового решения меньше; 2.Определенная вероятность принятия
        path = new_path
        path_dis = dis2
    else:
        path = old_path
        path_dis = dis1

    return path, path_dis

def picture(node_x, node_y, best_path):
    print('picture')
    """
    Нарисуйте функцию карты подъездных путей транспортного средства
    :параметр node_x: Местоположение хранилища по координате x
    :параметр node_y: Координата y местоположения магазина
    :параметр best_path: оптимальный путь
    :return:
    """

    def background():
        """
        Используйте plt для рисования фона системы координат, включая установку координат, размеров и другой информации
        """
        # font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)  # 从Извлеките китайские шрифты из системы, чтобы облегчить plt отображение китайских надписей
        plt.figure(figsize=(80, 80))  # Установите размер изображения (длина, ширина) в дюймах
        plt.xlim((0, 10))  # Установите диапазон отображения значений по оси x
        plt.ylim((0, 10))  # Установите диапазон отображения значений по оси y
        plt.xticks(np.linspace(0, 80, 11))  # Установите масштаб оси x, где 0 и 100 представляют левое и правое предельные значения оси x, а 11 представляет разделение оси x на 10 частей для разметки
        plt.yticks(np.linspace(0, 80, 11))  # Установите масштаб оси y, где 0 и 100 представляют левое и правое предельные значения оси y, а 11 представляет разделение оси y на 10 частей для разметки
        # plt.xlabel('x координата-схема', fontproperties=font)  # Установите метку, отображаемую на оси x, эффект будет китайским
        # plt.ylabel('y координата - схема', fontproperties=font)  # Установите метку, отображаемую на оси y, эффект будет китайским

    def random_color():
        """Генерируйте случайные шестнадцатеричные цвета и возвращайте"""
        colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
        color = ""  # Определите цвет в качестве строковых данных
        for i in range(6):
            color += colorArr[np.random.randint(0, 14)]
        return "#" + color

    x_list = []  # Определите координату узла x чертежа plt
    y_list = []  # Определите координату y узла, который рисует plt
    background()

    plt.scatter(node_x, node_y, c='r', s=50, alpha=1)  # Нарисуйте точечную диаграмму, то есть график магазина
    address_index = [i for i in range(len(best_path)) if best_path[i] == 0]  # Найдите координаты 0 (склад) в пути

    for i in range(len(address_index) - 1):
        for j in range(address_index[i], address_index[i + 1] + 1, 1):
            x_list.append(node_x[best_path[j]])
            y_list.append(node_y[best_path[j]])
        plt.plot(x_list, y_list, c=random_color())  # Нарисуйте траекторию движения каждого автомобиля
        x_list = []  # очищать
        y_list = []  # очищать

    plt.savefig('/home/julkinis/Документы/fqw/result.png')  # Сохраните визуальную карту пути доступа к транспортному средству
    plt.show()


def CVRP_SA(max_cars, Q, number_clients, path, kf):
    
    node_x, node_y, node_w = get_data(path)  # Считывайте координаты точек и запрашивайте информацию об объемах заказов
    
    # найти начальный путь алгоритмом CW        
    routes, costs = CW_alg(node_x, node_y, node_w, number_clients, Q)
    
    best_dist = 0
    init_path = []
    car = 0
    init_path.append(0)

    k = 1
    # преобразование путей в общий 
    for route in routes:
        if len(route.edges) > 1:
            best_dist += route.cost
            car += 1
            # print('Путь', k, 'автомобиля :')
            # print(route)
            # print('длина пути', round(route.cost, 2), 'объем груза', route.volume)
            k += 1
            for edge in route.edges:
                if car > max_cars-1 and edge.destination.id == 0:
                    break 
                init_path.append(edge.destination.id)
    init_path.append(0)
    
    
    if max_cars < car:
        print("Nевозможно решить CW для", max_cars, "машин")
    elif max_cars > car:
        for i in range(max_cars - car):
            init_path.append(0)

    T0, old_path = heat(init_path, node_w, max_cars, costs, Q)  # Начальная температура (путь после процесса нагрева используется в качестве начального пути)
        
        # max_cars = car
    print("\nРезультат работы алгоритма Кларка-Райта", round(best_dist, 2), '\n')
    # picture(node_x, node_y, init_path)
    # picture(node_x, node_y, init_path)

    # print('init_path', init_path, '\n')

    if max_cars >= car:
        old_path = init_path
    
    #print('old_path', old_path, '\n')
    # picture(node_x, node_y, old_path)
    #T_down_rate = 0.93  # Скорость падения температуры
    T_down_rate = kf
    T_end = 0.01  # Температура остановки
    K = 7000  # Порядок внутреннего цикла
    #K=kf
    count = math.ceil(math.log(T_end / T0, T_down_rate))  # Количество внешних циклов
    dis_T = np.zeros(count + 1)  # Оптимальное решение для каждого цикла
    best_path = init_path  # Установите оптимальный путь к исходному пути

    # best_path = old_path
    # old_path = init_path

    shortest_dis = np.inf  # Установите начальное оптимальное расстояние равным бесконечности
    n = 0
    T = T0  # Значение температуры в текущем цикле

    while T > T_end:  # Имитируйте отжиг до тех пор, пока температура не станет меньше температуры окончания
        for i in range(K):
            new_path = new_paths(old_path)  # Сгенерируйте новый путь
            
            old_path, path_dis = metropolis(old_path, new_path, node_w, max_cars, T, costs, Q)  # Определите текущий оптимальный путь по критерию мегаполиса
            if path_dis <= shortest_dis:  # Если оптимальный путь лучше результата предыдущего расчета, примите
                #print(new_path)
                shortest_dis = path_dis
                best_path = old_path

        dis_T[n] = shortest_dis  # Оптимальное решение для каждого цикла
        n += 1
        T *= T_down_rate  # После каждого цикла охлаждайте с определенной скоростью снижения
   
        # print(best_path)
        # print('best_dis', shortest_dis)  # Распечатав процесс выполнения, вы всегда сможете отслеживать значение T и текущее оптимальное расстояние
        # print(T)
    # print(dis_T)
    print('best_path', best_path)  # Выведите решение по оптимальному пути
    print('Результатом алгоритма отжига для вычисления задачи CVRP является', round(total_dis(best_path, node_w, max_cars, costs, Q), 2))  # Выведите оптимальный результат расчета, то есть сумму формальных траекторий движения транспортного средства
    address_index = [i for i in range(len(best_path)) if best_path[i] == 0]  # Найдите координаты 0 (склад) в пути
   
    for i in range(len(address_index) - 1):  # Выведите путь каждого автомобиля
        length = 0
        volume = 0
        print('Путь {} автомобиля равен：'.format(i + 1))
        print(best_path[address_index[i]:address_index[i + 1] + 1])

        for j in range(address_index[i], address_index[i + 1]):
            volume += node_w[best_path[j]]
            length += costs[best_path[j]][best_path[j+1]]
        print('path lengt', round(length, 2), ' volume', volume)
        
        

  #  picture(node_x, node_y, best_path)  # Визуализация пути подъезда транспортного средства

    return


if __name__ == '__main__':
    
    # start_time = datetime.datetime.now()
    # CVRP_SA(8, 200, 100, '/home/julkinis/Документы/fqw/Data.xls', '')
    # end_time = datetime.datetime.now()
    # print('Время выполнения программы равно：', end_time - start_time) 
    for i in range(90, 100, 1):
        print(i/100)
        start_time = datetime.datetime.now()
        CVRP_SA(7, 100, 44, '/home/julkinis/Документы/fqw/Data.xls', i/100)
        end_time = datetime.datetime.now()
        print('Время выполнения программы равно：', end_time - start_time) 
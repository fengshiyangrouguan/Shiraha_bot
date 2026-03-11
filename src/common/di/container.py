# src/system/di/container.py
from typing import Dict, Any, Type, Callable, TypeVar

T = TypeVar('T')

class Container:
    """
    一个简单的依赖注入容器，用于管理单例服务。
    """
    _instance = None
    _services: Dict[Type[Any], Any]
    _factories: Dict[Type[Any], Callable[[], Any]]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Container, cls).__new__(cls)
            cls._instance._services = {}
            cls._instance._factories = {}
        return cls._instance

    def register_instance(self, interface: Type[T], instance: T):
        """
        注册一个已经创建好的单例实例。

        Args:
            interface: 服务的类型或接口（通常是类名）。
            instance: 服务的单例实例。
        """
        try:
            if interface in self._services or interface in self._factories:
                raise Exception(
                    f"DI Container 注册失败：服务 {interface.__name__} 已经注册过，不能重复注册！"
                )

            print(f"DI Container: 注册实例 {interface.__name__}")
            self._services[interface] = instance

        except Exception as e:
            # 这里你可以将错误继续抛出，也可以记录日志
            raise e

    def register_factory(self, interface: Type[T], factory: Callable[[], T]):
        """
        注册一个用于创建单例服务的工厂函数。
        服务将在第一次被请求时创建。

        Args:
            interface: 服务的类型或接口。
            factory: 一个无参数并返回服务实例的函数。
        """
        print(f"DI Container: 注册工厂 {interface.__name__}")
        self._factories[interface] = factory

    def resolve(self, interface: Type[T]) -> T:
        """
        解析（获取）一个服务实例。

        Args:
            interface: 服务的类型或接口。

        Returns:
            服务的单例实例。
        
        Raises:
            Exception: 如果服务未被注册。
        """
        if interface not in self._services:
            if interface in self._factories:
                print(f"DI Container: 首次创建并缓存 {interface.__name__}")
                self._services[interface] = self._factories[interface]()
            else:
                raise Exception(f"服务 {interface.__name__} 未在 DI 容器中注册。")
        
        return self._services[interface]

# 创建一个全局唯一的容器实例，供整个应用使用
container = Container()

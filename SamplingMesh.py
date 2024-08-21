import multiprocessing as mp
import numpy as np
import typing
from copy import deepcopy
from time import time

NDIM = 4
DEBUG = False

class Pointer:
    def __init__(self, value):
        self.value = value
    
    def __str__(self):
        return self.value.__str__()

class TreeNode:
    # ""Enumeration""
    BUD, LEAF, BRANCH = 1, 2, 3

    def __init__(self, type, children, index = None): # Index is index array for bud
        self.type = type
        self.children = children
        self.index = index
        self.stable = False

    @classmethod
    def bud(cls, value, index):
        children = np.full((2,) * NDIM, None, dtype=object)
        children[tuple(index)] = value
        return cls(cls.BUD, children, index)

    @classmethod
    def branch(cls, children):
        return cls(cls.BRANCH, children)

    @classmethod
    def leaf(cls, values):
        return cls(cls.LEAF, values)

    @property
    def value(self):
        if self.type != TreeNode.BUD:
            raise RuntimeError("Cannot get nondescript value from non-BUD TreeNode")
        return self.children[tuple(self.index)]

    @value.setter
    def set_value(self, new_value: Pointer):
        if self.type != TreeNode.BUD:
            raise RuntimeError("Cannot set nondescript value from non-BUD TreeNode")
        self.children[tuple(self.index)] = new_value

    @staticmethod
    def index_array(index: int):
        return (np.floor(index / 2**np.arange(NDIM)) % 2).astype(int)

    # pivot is the center of a node; scale is the side length
    @staticmethod
    def child_pivot_and_scale(pivot: np.ndarray, scale: np.ndarray, index: np.ndarray):
        return pivot + scale * (2 * index - 1)/4, scale/2

    @staticmethod
    def parent_pivot_and_scale(pivot: np.ndarray, scale: np.ndarray, index: np.ndarray):
        return pivot - scale * (2 * index - 1)/2, scale * 2

    @staticmethod
    def leaf_location(pivot: np.ndarray, scale: np.ndarray, index: np.ndarray):
        return pivot + scale/2 * (2 * index - 1)

    @staticmethod
    def normalize_location(pivot: np.ndarray, scale: np.ndarray, location: np.ndarray):
        return (location - pivot)/scale * 2

    def updated(self, value: Pointer = None):
        if value is None:
            value = self.value
        cur = self.type
        return lambda: (self.type != cur) or (not np.isnan(value.value))

    def recfind(self, location: np.ndarray, pivot: np.ndarray, scale: np.ndarray) -> Pointer:
        if DEBUG:
            if not np.all(self.pivot == pivot):
                print("Pivots don't match")
                print(pivot)
                print(self.pivot)
            if not np.all(self.scale == scale):
                print("Scales don't match")
                print(scale)
                print(self.scale)
        if self.type == TreeNode.BRANCH:
            close = np.isclose(pivot, location, atol=1e-10, rtol=0)
            greater = location > pivot
            for i in range(2 ** NDIM):
                index_array = self.index_array(i)
                if np.all(np.logical_or(index_array.astype(bool) == greater, close)):
                    res = self.children[tuple(index_array)].recfind(location, *self.child_pivot_and_scale(pivot, scale, index_array))
                    if res is not None:
                        return res
            return None
        elif self.type == TreeNode.LEAF:
            lower_close = np.isclose(location, self.leaf_location(pivot, scale, np.zeros(NDIM)), atol=1e-10, rtol=0).astype(int)
            upper_close = np.isclose(location, self.leaf_location(pivot, scale, np.ones(NDIM)), atol=1e-10, rtol=0).astype(int)
            if np.any(lower_close + upper_close == 0):
                return None
            else:
                return self.children[tuple(upper_close)]
        elif self.type == TreeNode.BUD:
            index_array = self.index_array(self.index)
            if np.allclose(location, self.leaf_location(pivot, scale, index_array), atol=1e-10, rtol=0):
                return self.children[tuple(index_array)]
            else:
                return None
        else:
            raise ValueError("TreeNode.type must be TreeNode.BRANCH, TreeNode.LEAF, or TreeNode.BUD")

    def corner_value(self, corner: np.ndarray) -> Pointer:
        if self.type == TreeNode.BRANCH:
            return self.children[tuple(corner)].corner_value(corner)
        if self.type == TreeNode.LEAF:
            return self.children[tuple(corner)]
        if self.type == TreeNode.BUD:
            assert(np.all(self.index == corner))
            return self.value

        raise ValueError("TreeNode.type must be TreeNode.BRANCH, TreeNode.LEAF, or TreeNode.BUD")

    def adjacent_corner_values(self, corner: np.ndarray) -> list[Pointer]:
        output = []
        corner_copy = corner.copy()
        for i in range(NDIM):
            corner_copy[i] = 1 if corner_copy[i] == 0 else 0
            output.append(self.corner_value(corner_copy))
            corner_copy[i] = corner[i]
        return output

    def print(self, depth: int = 0):
        out = "\t" * depth
        if self.type == TreeNode.BRANCH:
            print(out + "BRANCH: ")
            for child in self.children.flatten():
                child.print(depth + 1)
        elif self.type == TreeNode.LEAF:
            out += "LEAF: "
            for child in self.children.flatten():
                out += str(child) + " "
            print(out)
        else:
            assert(self.type == TreeNode.BUD)
            print(out + "BUD: " + str(self.value))


class OngoingTask:
    PENDING, WAITING, COMPLETE = 1, 2, 3

    def __init__(self, state, calculations = None, destinations = None, condition = None, next_task = None, value = None):
        self.state = state
        self.calculations = calculations
        self.destinations = destinations
        self.condition = condition
        self.next_task = next_task
        self.value = value

    @classmethod
    def pending_task(cls, calculations: list, destinations: list[Pointer], next_task):
        assert(len(calculations) == len(destinations))
        return cls(OngoingTask.PENDING, calculations = calculations, destinations = destinations, next_task = next_task)

    @classmethod
    def waiting_task(cls, condition, next_task):
        return cls(OngoingTask.WAITING, condition = condition, next_task = next_task)

    @classmethod
    def complete_task(cls, value):
        return cls(OngoingTask.COMPLETE, value = value)

    def complete(self):
        return self.state == OngoingTask.COMPLETE

    def ready(self) -> bool:
        if self.state == OngoingTask.PENDING:
            for calc in self.calculations:
                if not calc.ready():
                    return False
            return True
        elif self.state == OngoingTask.WAITING:
            return self.condition()
        else:
            raise RuntimeError("Cannot call ready on COMPLETE task")

    def next(self):
        assert(self.state != OngoingTask.COMPLETE)
        if self.state == OngoingTask.PENDING:
            for i in range(len(self.calculations)):
                self.destinations[i].value = self.calculations[i].get()
        return self.next_task()

class SamplingMesh:
    def __init__(self, func, pool, atol = 0, rtol = 0, pivot = None, scale = None, args = None, kwargs = None):
        if atol == 0 and rtol == 0:
            raise ValueError("Must have nonzero error tolerance")
        self.func = func
        self.pool = pool
        self.atol = atol
        self.rtol = rtol
        self.pivot = pivot if pivot is not None else np.zeros(NDIM)
        self.scale = scale if scale is not None else np.ones(NDIM)
        self.max_leaf_scale = np.copy(self.scale)
        self.root = None
        self.args = args if args is not None else tuple()
        self.kwargs = kwargs if kwargs is not None else {}
        self.func_calls = 0
        self.interpolations = 0
        self.records = {}

    # Position is normalized to have interpolating points at +/- 1 in all dimensions
    # Generalized form of https://en.wikipedia.org/wiki/Bilinear_interpolation
    @staticmethod
    def interpolate(position: np.ndarray, data: np.ndarray) -> float:
        interpolated = data
        for i in range(NDIM):
            interpolated = np.array([1 - position[i], position[i] + 1]) @ interpolated
        return interpolated / 2**NDIM

    def submit_func(self, pivot: np.ndarray, scale: np.ndarray, index: np.ndarray):
        self.func_calls += 1
        return self.pool.apply_async(self.func, (TreeNode.leaf_location(pivot, scale, index), *self.args), self.kwargs)

    def do_interpolation(self, location: np.ndarray, pivot: np.ndarray, scale: np.ndarray, node: TreeNode) -> OngoingTask:
        if DEBUG:
            self.records[tuple(location)].append("do_interpolation")
        if node.type == TreeNode.BRANCH:
            return self.descend_tree(location, pivot, scale, node)
        elif node.type == TreeNode.BUD:
            raise ValueError("Cannot interpolate BUD")

        if not node.stable:
            return OngoingTask.waiting_task(lambda: node.stable, lambda: self.do_interpolation(location, pivot, scale, node))

        data = np.empty((2,) * NDIM)
        for i in range(2 ** NDIM):
            index_array = node.index_array(i)
            pointer = node.children[tuple(index_array)]
            assert(isinstance(pointer, Pointer))
            if np.isnan(pointer.value):
                return OngoingTask.waiting_task(node.updated(pointer), lambda: self.do_interpolation(location, pivot, scale, node))
            data[tuple(index_array)] = pointer.value
        self.interpolations += 1
        return OngoingTask.complete_task(self.interpolate(node.normalize_location(pivot, scale, location), data))

    # scale and pivot are of child, not parent
    def zoom_in(self, location: np.ndarray, pivot: np.ndarray, scale: np.ndarray, parent: TreeNode, corner: np.ndarray) -> OngoingTask:
        if DEBUG:
            self.records[tuple(location)].append("zoom_in")
        child = parent.children[tuple(corner)]
        if child.type == TreeNode.BRANCH:
            return self.descend_tree(location, pivot, scale, child)
        assert(child.type == TreeNode.LEAF)

        corner_copy = corner.copy()
        unscaled_curvatures = np.zeros(NDIM)
        for i in range(NDIM):
            corner_copy[i] = 1 if corner_copy[i] == 0 else 0
            p1 = child.children[tuple(corner)]
            if np.isnan(p1.value):
                return OngoingTask.waiting_task(lambda: not np.isnan(p1.value), lambda: self.zoom_in(location, pivot, scale, parent, corner))
            p2 = child.children[tuple(corner_copy)]
            if np.isnan(p2.value):
                return OngoingTask.waiting_task(lambda: not np.isnan(p2.value), lambda: self.zoom_in(location, pivot, scale, parent, corner))
            p3 = parent.corner_value(corner_copy)
            if np.isnan(p3.value):
                return OngoingTask.waiting_task(lambda: not np.isnan(p3.value), lambda: self.zoom_in(location, pivot, scale, parent, corner))
            unscaled_curvatures[i] = p1.value - 2 * p2.value + p3.value
            corner_copy[i] = corner[i]
        
        mse = 0
        for i in range(NDIM):
            mse += unscaled_curvatures[i]**2 * 31 / 120
            for j in range(i + 1, NDIM):
                mse += unscaled_curvatures[i]**2 * unscaled_curvatures[j]**2 * 25 / 144
        
        assert(not np.isnan(mse))
        
        if mse > (self.atol + self.rtol * np.abs(np.mean([x.value for x in child.children.flatten()]) or np.any(scale > self.max_leaf_scale))) ** 2:
            if child.stable:
                parent.print()
                raise RuntimeError("Shouldn't exceed error on stable leaf")
            # Convert the child from a LEAF into a BRANCH with BUD children
            child.type = TreeNode.BRANCH
            child.stable = True
            if DEBUG:
                child.pivot = np.copy(pivot)
                child.scale = np.copy(scale)
            for i in range(2 ** NDIM):
                index = child.index_array(i)
                child.children[tuple(index)] = TreeNode.bud(child.children[tuple(index)], index)
                if DEBUG:
                    node = child.children[tuple(index)]
                    node.pivot, node.scale = child.child_pivot_and_scale(pivot, scale, index)
            corner = (location > pivot).astype(int)
            return self.bud_to_leaf(location, *child.child_pivot_and_scale(pivot, scale, corner), child, corner)
        else:
            child.stable = True
            return self.do_interpolation(location, pivot, scale, child)

    # pivot and scale are of the leaf-to-be, not the parent
    def bud_to_leaf(self, location: np.ndarray, pivot: np.ndarray, scale: np.ndarray, parent: TreeNode, index: np.ndarray) -> OngoingTask:
        if DEBUG:
            self.records[tuple(location)].append("bud_to_leaf")
        bud = parent.children[tuple(index)]

        if bud.type == TreeNode.LEAF:
            # # If this has become a leaf since this being queued, someoneone else can handle the zoom_in
            # if bud.stable:
            #     return self.do_interpolation(location, pivot, scale, bud)
            # else:
            #     return OngoingTask.waiting_task(lambda: bud.stable, lambda: self.do_interpolation(location, pivot, scale, bud))
            return self.zoom_in(location, pivot, scale, parent, index)
        if bud.type == TreeNode.BRANCH:
            return self.descend_tree(location, pivot, scale, bud)
        
        if np.isnan(bud.value.value):
            return OngoingTask.waiting_task(bud.updated(), lambda: self.bud_to_leaf(location, pivot, scale, parent, index))

        bud.type = TreeNode.LEAF
        destinations = []
        calculations = []
        waiting = []
        for i in range(2 ** NDIM):
            cur_index = bud.index_array(i)
            if np.any(cur_index != index):
                res = self.root.recfind(bud.leaf_location(pivot, scale, cur_index), self.pivot, self.scale)
                if res is None:
                    if DEBUG:
                        self.records[tuple(location)].append(f"recfind failed to find for {cur_index}")
                    bud.children[tuple(cur_index)] = Pointer(np.NaN)
                    destinations.append(bud.children[tuple(cur_index)])
                    calculations.append(self.submit_func(pivot, scale, cur_index))
                else:
                    if DEBUG:
                        self.records[tuple(location)].append(f"recfind found {res.value}; saving to {cur_index}")
                    bud.children[tuple(cur_index)] = res
                    if np.isnan(res.value):
                        waiting.append(res)
        if len(waiting) > 0:
            next_task = OngoingTask.waiting_task(lambda: not np.any(np.isnan([x.value for x in waiting])), lambda: self.zoom_in(location, pivot, scale, parent, index))
            return OngoingTask.pending_task(calculations, destinations, lambda: next_task)
        elif len(calculations) > 0:
            return OngoingTask.pending_task(calculations, destinations, lambda: self.zoom_in(location, pivot, scale, parent, index))
        else:
            return self.zoom_in(location, pivot, scale, parent, index)

    def descend_tree(self, location: np.ndarray, pivot: np.ndarray, scale: np.ndarray, node: TreeNode) -> OngoingTask:
        if DEBUG:
            self.records[tuple(location)].append("descend_tree")
        prev = None
        while node.type == TreeNode.BRANCH:
            index = (location > pivot).astype(int)
            pivot, scale = node.child_pivot_and_scale(pivot, scale, index)
            prev = node
            node = node.children[tuple(index)]

        if node.type == TreeNode.BUD:
            if prev is None:
                raise ValueError("Cannot call SamplingMesh.descend_tree on a BUD")
            return self.bud_to_leaf(location, pivot, scale, prev, index)
        else:
            assert(node.type == TreeNode.LEAF)
            return self.do_interpolation(location, pivot, scale, node)

    def expand_tree(self, location: np.ndarray) -> OngoingTask:
        if DEBUG:
            if tuple(location) not in self.records:
                self.records[tuple(location)] = []
            self.records[tuple(location)].append("expand_tree")
        too_low = location < TreeNode.leaf_location(self.pivot, self.scale, np.zeros(NDIM))
        too_high = location > TreeNode.leaf_location(self.pivot, self.scale, np.ones(NDIM))
        if np.any(too_low) or np.any(too_high) or self.root is None:
            # Expand in positive direction by default
            too_low = too_low.astype(int)
            self.pivot, self.scale = TreeNode.parent_pivot_and_scale(self.pivot, self.scale, too_low)
            children = np.empty((2,) * NDIM, dtype=object)
            destinations = []
            calculations = []
            for i in range(2 ** NDIM):
                index = TreeNode.index_array(i)
                if np.all(index == too_low) and self.root is not None:
                    children[tuple(index)] = self.root
                else:
                    dest = Pointer(np.NaN)
                    children[tuple(index)] = TreeNode.bud(dest, index)
                    if DEBUG:
                        node = children[tuple(index)]
                        node.pivot, node.scale = TreeNode.child_pivot_and_scale(self.pivot, self.scale, index)
                    destinations.append(dest)
                    calculations.append(self.submit_func(self.pivot, self.scale, index))
            self.root = TreeNode.branch(children)
            if DEBUG:
                self.root.pivot = np.copy(self.pivot)
                self.root.scale = np.copy(self.scale)
            return OngoingTask.pending_task(calculations, destinations, lambda: self.expand_tree(location))
        else:
            return self.descend_tree(location, self.pivot, self.scale, self.root)

    def validate_tree(self, pivot = None, scale = None, node = None) -> bool:
        if self.root is None:
            return True
        if node is None:
            node = self.root
        if pivot is None:
            pivot = self.pivot
        if scale is None:
            scale = self.scale
        
        if DEBUG:
            if np.any(pivot != node.pivot):
                print("Wrong pivot")
                print(pivot)
                print(node.pivot)
                return False
            if np.any(scale != node.scale):
                print("Wrong scale")
                print(scale)
                print(node.scale)
                return False
        
        if node.type == TreeNode.BRANCH:
            for i in range(2 ** NDIM):
                index_array = TreeNode.index_array(i)
                if not self.validate_tree(*TreeNode.child_pivot_and_scale(pivot, scale, index_array), node.children[tuple(index_array)]):
                    return False
            return True
        elif node.type == TreeNode.LEAF:
            for i in range(2 ** NDIM):
                index_array = TreeNode.index_array(i)
                if np.isnan(node.children[tuple(index_array)].value):
                    continue
                if not np.isclose(self.func(TreeNode.leaf_location(pivot, scale, index_array)), node.children[tuple(index_array)].value, atol=1e-10, rtol=0):
                    print("Wrong leaf value")
                    print(self.func(TreeNode.leaf_location(pivot, scale, index_array)))
                    print(node.children[tuple(index_array)].value)
                    return False
            return True
        else:
            if np.isnan(node.value.value):
                return True
            if np.isclose(self.func(TreeNode.leaf_location(pivot, scale, node.index)), node.value.value, atol=1e-10, rtol=0):
                return True
            else:
                print("Wrong bud value")
                print(self.func(TreeNode.leaf_location(pivot, scale, node.index)))
                print(node.value.value)
                return False

    def multi_interpolate(self, locations, max_time = np.inf):
        start = time()
        tasks = [self.expand_tree(loc) for loc in locations]
        results = np.ones(len(tasks)) * np.NaN

        incomplete = True
        while incomplete:
            if time() - start > max_time:
                break
            incomplete = False
            for i in range(locations.shape[0]):
                if tasks[i] is None:
                    continue
                if tasks[i].complete():
                    results[i] = tasks[i].value
                    tasks[i] = None
                else:
                    incomplete = True
                    if tasks[i].ready():
                        tasks[i] = tasks[i].next()
                if DEBUG:
                    try:
                        res = self.validate_tree()
                    except:
                        res = False
                    if not res:
                        print(self.records[tuple(locations[i,:])])
                        self.print()
                        raise RuntimeError("Tree has become invalid")
        return results

    def print(self):
        print(f"Pivot: {self.pivot}")
        print(f"Scale: {self.scale}")
        print(f"Absolute tolerance: {self.atol}")
        print(f"Relative tolerance: {self.rtol}")
        self.root.print()

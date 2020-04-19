# array

## 283 move-zeroes

### python



```python
# in-place
def moveZeroes(self, nums):
    zero = 0  # records the position of "0"
    for i in xrange(len(nums)):
        if nums[i] != 0:
            nums[i], nums[zero] = nums[zero], nums[i]
            zero += 1
```



```python
def moveZeroes(self, nums):
	for i in range(len(nums))[::-1]:
        if nums[i] == 0:
            nums.pop(i)
            nums.append(0)
```



```python
def moveZeroes(nums):
    j=0
    for i in range(len(nums)):
       if nums[i] !=0:
           nums[j]=nums[i]
           j+=1
    while j<len(nums):
        nums[j]=0
        j+=1
```



```python
def moveZeroes(nums):
    nums.sort(key=lambda x: x == 0)
```

### Go



```go
func moveZeroes(nums []int) {

	index := 0

	for i := 0; i < len(nums); i++ {
		if nums[i] != 0 {
			nums[index] = nums[i]
			index++
		}
	}
	for i := index; i < len(nums); i++ {
		nums[i] = 0
	}
}
```



```go
func moveZeroes(nums []int)  {
    b := nums[:0]
	lend := 0
	for _, x := range nums {
		if x!=0 {
			b = append(b, x)	
		} else {
			lend++
		}	
	}
	for lend > 0 {
		b = append(b, 0)
		lend--
	}
}
```



```go
func moveZeroes(nums []int) {
	for i, j := 0, 0; j < len(nums); {
		for ; i < len(nums) && nums[i] != 0; i++ {}
		for ; j < len(nums) && (i >= j || nums[j] == 0); j++ {}
		if j < len(nums) {
			nums[i], nums[j] = nums[j], nums[i]
		}
	}
}
```



```go
func moveZeroes(nums []int)  {

	slow,fast:=0,0
	for fast<len(nums){
		if nums[fast]!=0 {
			nums[slow],nums[fast]=nums[fast],nums[slow]
			slow++
		}
		fast++
	}
}
```



### C++

```c++
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int j = 0;
        // move all the nonzero elements advance
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] != 0) {
                nums[j++] = nums[i];
            }
        }
        for (;j < nums.size(); j++) {
            nums[j] = 0;
        }
    }
};
```



```c++
void moveZeroes(vector<int>& nums) {
    int last = 0, cur = 0;
    
    while(cur < nums.size()) {
        if(nums[cur] != 0) {
            swap(nums[last], nums[cur]);
            last++;
        }
        
        cur++;
    }
}
```



```c++
void moveZeroes(vector<int>& nums) {
  for (int i = 0, j = 0; i < nums.size(); i++) if(nums[i] != 0) swap(nums[i], nums[j++]);
}

 void moveZeroes(vector<int>& nums) {
    for (int i = 0, j = 0; i < nums.size(); i++)  {
        if (nums[i] != 0) swap(nums[i], nums[j++]);
    }
}
```



## 11 container-with-most-water

### python



```python
def maxArea(self, height):
    left, right, maxWater, minHigh = 0, len(height) - 1, 0, 0
    while left < right:
        if height[left] < height[right]:
            minHigh, left = height[left], left+1
        else:
            minHigh,right = height[right], right-1
        maxWater = max(maxWater, (right - left + 1) * minHigh)
    return maxWater
```



### go

```python
func maxArea(height []int) int {
    mx := 0
    i, j := 0, len(height) - 1
    var vol int
    for i < j {
        if height[i] > height[j] {
            vol = (j - i) * height[j]
            j--
        } else {
            vol = (j - i) * height[i]
            i++
        }
        if mx < vol {
            mx = vol
        }
    }
    return mx
}
```



### C++

```c++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int i=0,j=height.size()-1,ans = 0;
        while(j>i)
        {
            ans = max(min(height[i],height[j])*(j-i),ans);
            if(height[i]>height[j]) j--;
            else i++;
        }
        return ans;
    }
};
```



```c++
class Solution {
public:
    int maxArea(vector<int>& height) {
        int ans = 0;
        int i = 0, j = height.size() - 1;          
        while(i < j){
            ans = max(ans, (j - i) * min(height[i], height[j]));
            height[i] > height[j] ? j-- : i++;  
        }
        
        return ans;
    }
};
```



## 70 climbing-stairs

 

### python

```python
# Top down - TLE
def climbStairs1(self, n):
    if n == 1:
        return 1
    if n == 2:
        return 2
    return self.climbStairs(n-1)+self.climbStairs(n-2)
 
# Bottom up, O(n) space
def climbStairs2(self, n):
    if n == 1:
        return 1
    res = [0 for i in xrange(n)]
    res[0], res[1] = 1, 2
    for i in xrange(2, n):
        res[i] = res[i-1] + res[i-2]
    return res[-1]

# Bottom up, constant space
def climbStairs3(self, n):
    if n == 1:
        return 1
    a, b = 1, 2
    for i in xrange(2, n):
        tmp = b
        b = a+b
        a = tmp
    return b
    
# Top down + memorization (list)
def climbStairs4(self, n):
    if n == 1:
        return 1
    dic = [-1 for i in xrange(n)]
    dic[0], dic[1] = 1, 2
    return self.helper(n-1, dic)
    
def helper(self, n, dic):
    if dic[n] < 0:
        dic[n] = self.helper(n-1, dic)+self.helper(n-2, dic)
    return dic[n]
    
# Top down + memorization (dictionary)  
def __init__(self):
    self.dic = {1:1, 2:2}
    
def climbStairs(self, n):
    if n not in self.dic:
        self.dic[n] = self.climbStairs(n-1) + self.climbStairs(n-2)
    return self.dic[n]
```



```python
def climbStairs(self, n):
    a, b = 1, 1
    for i in range(n):
        a, b = b, a + b
    return a
```

### go

```python
func climbStairs(n int) int {
    a,b:=1,1
    for i:=0;i<n;i++{
        a,b=b,a+b
    }
    return a
}
```



```go
func climbStairs(n int) int {
    curr := 1
    a := 1
    b := 1
    for i := 2; i <= n; i++ {
        curr = a + b
        a = b
        b = curr
    }
    return curr
}
```

```go
func climbStairs(n int) int {
    res := make([]int, n+1)
    res[0] = 1
    res[1] = 1
    for i := 2; i <= n; i++ {
        res[i] = res[i-1] + res[i-2]
    }
    return res[n]
}
```

### C++

```c++
class Solution {
public:
    int climbStairs(int n) {
        if (n==1) return 1;
        if (n==2) return 2;
        vector<int> step(n,0);
        step[0]=1;
        step[1]=2;
        for (int i=2;i<n;i++){
            step[i]=step[i-1]+step[i-2];
        }
        return step[n-1];
    }
};
```



```c++
class Solution {
public:
    int climbStairs(int n) {
        int StepOne = 1;
        int StepTwo = 0;
        int ret = 0;
        for(int i=0;i<n;i++)
        {
            ret = StepOne + StepTwo;
            StepTwo = StepOne;
            StepOne = ret;
        }
        return ret;
    }
};
```

## 1 two sum

### python

```python
class Solution(object):
    def twoSum(self, nums, target):        
        dic={}
        for i in enumerate(nums):
            if dic.get(target-nums):
                return [i,dic.get(target-nums)]
            dic[nums]=i
```

```python
class Solution(object):
    def twoSum(self, nums, target):
        if len(nums) <= 1:
            return False
        buff_dict = {}
        for i in range(len(nums)):
            if nums[i] in buff_dict:
                return [buff_dict[nums[i]], i]
            else:
                buff_dict[target - nums[i]] = i
```

### go

```python
func twoSum(nums []int, target int) []int {
    m := make(map[int]int)
    for i, n := range nums {
        _, prs := m[n]
        if prs {
            return []int{m[n], i}
        } else {
            m[target-n] = i
        }
    }
    return nil
}
```



```go
func twoSum(nums []int, target int) []int {
        tmpMap := make(map[int]int)
        for i, num := range nums {
                if _, ok := tmpMap[target-num]; ok {
                        return []int{tmpMap[target-num], i}
                }
                tmpMap[num] = i
        }
        return []int{}

```



### C++

```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> indices;
        for (int i = 0; i < nums.size(); i++) {
            if (indices.find(target - nums[i]) != indices.end()) {
                return {indices[target - nums[i]], i};
            }
            indices[nums[i]] = i;
        }
        return {};
    }
}
```



```c++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        std::unordered_map<int, std::size_t> tmp;
    for (std::size_t i = 0; i < nums.size(); ++i) {
      if (tmp.count(target - nums[i])) {
        return {tmp[target - nums[i]], i};
      }
      tmp[nums[i]] = i;
    }
    return {nums.size(), nums.size()};
    }
};
```



## 05 3sum

### python

```python
def threeSum(self, nums):
    res = []
    nums.sort()
    for i in range(len(nums)-2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        l, r = i+1, len(nums)-1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l +=1 
            elif s > 0:
                r -= 1
            else:
                res.append((nums[i], nums[l], nums[r]))
                while l < r and nums[l] == nums[l+1]:
                    l += 1
                while l < r and nums[r] == nums[r-1]:
                    r -= 1
                l += 1; r -= 1
    return res
```

```python
class Solution(object):
    def threeSum(self, nums):
        if len(nums) <3:
            return []
        nums.sort()
        res=set()
        for i,v in enumerate(nums[:-2]):
            if i >1 and v==nums[i-1]:
                continue
            d={}
            for x in nums[i+1:]:
                if x not in d:
                    d[-v-x]=1
                else:
                    res.add((v,-v-x,x))
        return map(list,res)
```



### go

```go
func threeSum(nums []int) [][]int {
    sort.Ints(nums)
	var array [][]int
	if nums==nil || len(nums)<3{
		return array
	}
	for i:=0;i<len(nums)-2;i++{
		if nums[i]>0{
			return array
		}
		if i>0 && nums[i]==nums[i-1]{
			continue
		}
		l:=i+1
		r:=len(nums)-1
		for l<r {
			sum:=nums[i]+nums[l]+nums[r]
			if sum >0{
				r--
                continue
			}
			if sum<0{
				l++
                continue
			}
			if sum==0{
				array = append(array, []int{nums[i],nums[l],nums[r]})
				for l<r && nums[l]==nums[l+1] {
					l++
				}
				for l<r && nums[r]==nums[r-1] {
						r--
				}
			}
			l++
			r--
		}
	}
	return array
}
```



### C++

```c++
vector<vector<int>> threeSum(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> res;
    for (unsigned int i=0; i<nums.size(); i++) {
        if ((i>0) && (nums[i]==nums[i-1]))
            continue;
        int l = i+1, r = nums.size()-1;
        while (l<r) {
            int s = nums[i]+nums[l]+nums[r];
            if (s>0) r--;
            else if (s<0) l++;
            else {
                res.push_back(vector<int> {nums[i], nums[l], nums[r]});
                while (nums[l]==nums[l+1]) l++;
                while (nums[r]==nums[r-1]) r--;
                l++; r--;
            }
        }
    }
    return res;
}
```

======

## 06 remove-duplicates-from-sorted-array

```python
class Solution(object):
    def removeDuplicates(self, nums):
        if not nums:
            return
        i=0
        for j in range(len(nums)):
            if nums[j] !=nums[i]:
                i+=1
                nums[i]=nums[j]
        return i+1
```





```python
class Solution:
    def removeDuplicates(self, nums):
        if not nums:
            return 
        slow = fast = 0
        while fast <= len(nums) - 1:
            if nums[fast] != nums[slow]:
                nums[slow+1] = nums[fast]
                slow += 1
            fast += 1
        return slow + 1
```



```python
from collections import OrderedDict
class Solution(object):
    def removeDuplicates(self, nums):

        nums[:] =  OrderedDict.fromkeys(nums).keys()
        return len(nums)
```

## 07 remove-element

### python

```python
def removeElement(self, nums, val):
    i = 0
    for x in nums:
        if x != val:
            nums[i] = x
            i += 1
    return i
```



```python
def removeElement(self, nums, val):
    try:
        while True:
            nums.remove(val)
    except:
        return len(nums)
```



```python
class Solution(object):
    def removeElement(self, nums, val):
        for x in nums[:]:
            if x == val:
                nums.remove(val)
        return len(nums)
```



## 08search-insert-position

### python

```python
class Solution(object):
    def searchInsert(self, nums, target):   
        return len([x for x in nums if x<target])
```



```python
def searchInsert(self, nums, target):
    l , r = 0, len(nums)-1
    while l <= r:
        mid=(l+r)/2
        if nums[mid]== target:
            return mid
        if nums[mid] < target:
            l = mid+1
        else:
            r = mid-1
    return l
```



## 09 maximum-subarray

### python

```python
class Solution(object):
    def maxSubArray(self, nums):
        for i in range(1, len(nums)):
            if nums[i-1] > 0:
                nums[i] += nums[i-1]
        return max(nums)
```



```python
class Solution(object):
    def maxSubArray(self, nums):
        if not nums:
            return 0
        curSum=maxSum=nums[0]
        for num in nums[1:]:
            curSum=max(num,curSum+num)
            maxSum=max(maxSum,curSum)
        return maxSum
```



```python
class Solution(object):
    def maxSubArray(self, nums):
        for i in xrange(1,len(nums)):nums[i]=max(nums[i], nums[i]+nums[i-1])
        return max(nums)
```

## 10 3sum-closest

### python

```python
class Solution(object):
    def threeSumClosest(self, nums, target):
        res = []
        nums.sort()
        result=nums[0]+nums[1]+nums[2]
        for i in range(len(nums)-2):
            l, r = i+1, len(nums)-1
            while l < r:
                sum = nums[i] + nums[l] + nums[r]
                if sum ==target:
                    return sum
                if abs(sum-target)<abs(result-target):
                    result=sum
                if sum < target:
                    l +=1 
                else:
                    r-=1
        return result
```



## 11 4sum

### python

```python
def fourSum(self, nums, target):
    nums.sort()
    results = []
    self.findNsum(nums, target, 4, [], results)
    return results

def findNsum(self, nums, target, N, result, results):
    if len(nums) < N or N < 2: return

    # solve 2-sum
    if N == 2:
        l,r = 0,len(nums)-1
        while l < r:
            if nums[l] + nums[r] == target:
                results.append(result + [nums[l], nums[r]])
                l += 1
                r -= 1
                while l < r and nums[l] == nums[l - 1]:
                    l += 1
                while r > l and nums[r] == nums[r + 1]:
                    r -= 1
            elif nums[l] + nums[r] < target:
                l += 1
            else:
                r -= 1
    else:
        for i in range(0, len(nums)-N+1):   # careful about range
            if target < nums[i]*N or target > nums[-1]*N:  # take advantages of sorted list
                break
            if i == 0 or i > 0 and nums[i-1] != nums[i]:  # recursively reduce N
                self.findNsum(nums[i+1:], target-nums[i], N-1, result+[nums[i]], results)
    return
```



```python
class Solution:
# @return a list of lists of length 4, [[val1,val2,val3,val4]]
def fourSum(self, num, target):
    two_sum = collections.defaultdict(list)
    res = set()
    for (n1, i1), (n2, i2) in itertools.combinations(enumerate(num), 2):
        two_sum[i1+i2].append({n1, n2})
    for t in list(two_sum.keys()):
        if not two_sum[target-t]:
            continue
        for pair1 in two_sum[t]:
            for pair2 in two_sum[target-t]:
                if pair1.isdisjoint(pair2):
                    res.add(tuple(sorted(num[i] for i in pair1 | pair2)))
        del two_sum[t]
    return [list(r) for r in res]
```



```python
class Solution:
    # @return a list of lists of length 4, [[val1,val2,val3,val4]]
    def fourSum(self, num, target):
        num.sort()
        result = []
        for i in xrange(len(num)-3):
            if num[i] > target/4.0:
                break
            if i > 0 and num[i] == num[i-1]:
                continue
            target2 = target - num[i]
            for j in xrange(i+1, len(num)-2):
                if num[j] > target2/3.0:
                    break
                if j > i+1 and num[j] == num[j-1]:
                    continue
                k = j + 1
                l = len(num) - 1
                target3 = target2 - num[j]
                # we should use continue not break
                # because target3 changes as j changes
                if num[k] > target3/2.0:
                    continue
                if num[l] < target3/2.0:
                    continue
                while k < l:
                    sum_value = num[k] + num[l]
                    if sum_value == target3:
                        result.append([num[i], num[j], num[k], num[l]])
                        kk = num[k]
                        k += 1
                        while k<l and num[k] == kk:
                            k += 1
                        
                        ll = num[l]
                        l -= 1
                        while k<l and num[l] == ll:
                            l -= 1
                    elif sum_value < target3:
                        k += 1
                    else:
                        l -= 1
        return result
```



```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        result = []
        if not nums or len(nums) < 4:
            return result
        nums.sort()
        length = len(nums)
        for k in range(length - 3):
            if k > 0 and nums[k] == nums[k - 1]:
                continue
            min1 = nums[k] + nums[k+1] + nums[k+2] + nums[k+3]
            if min1 > target:
                break
            max1 = nums[k] + nums [length-1] + nums[length - 2] + nums[length - 3]
            if max1 < target:
                continue
            for i in range(k+1, length-2):
                if i > k + 1 and nums[i] == nums[i - 1]:
                    continue
                j = i + 1
                h = length - 1
                min2 = nums[k] + nums[i] + nums[j] + nums[j + 1]
                if min2 > target:
                    continue
                max2 = nums[k] + nums[i] + nums[h] + nums[h - 1]
                if max2 < target:
                    continue
                while j < h:
                    curr = nums[k] + nums[i] + nums[j] + nums[h]
                    if curr == target:
                        result.append([nums[k], nums[i], nums[j], nums[h]])
                        j += 1
                        while j < h and nums[j] == nums[j - 1]:
                            j += 1
                        h -= 1
                        while j < h and i < h and nums[h] == nums[h + 1]:
                            h -= 1
                    elif curr > target:
                        h -= 1
                    elif curr < target:
                        j += 1
        return result
```

## 12next-permutation

### python

```python
def nextPermutation(self, nums):
    i = j = len(nums)-1
    while i > 0 and nums[i-1] >= nums[i]:
        i -= 1
    if i == 0:   # nums are in descending order
        nums.reverse()
        return 
    k = i - 1    # find the last "ascending" position
    while nums[j] <= nums[k]:
        j -= 1
    nums[k], nums[j] = nums[j], nums[k]  
    l, r = k+1, len(nums)-1  # reverse the second part
    while l < r:
        nums[l], nums[r] = nums[r], nums[l]
        l +=1 ; r -= 1
```

## 13 search-in-rotated-sorted-array

### python

```python
class Solution(object):
    def search(self, nums, target):
        if not nums:
            return -1
        left,right=0,len(nums)-1
        while left <= right:
            mid=(left+right)/2
            if target == nums[mid]:
                return mid
            if nums[left] <=nums[mid]:
                if nums[left] <= target<= nums[mid]:
                    right= mid - 1
                else:
                    left= mid + 1
            else:
                if nums[mid] <= target <= nums[right]:
                    left =mid + 1
                else:
                    right= mid - 1
        return -1
```



## 14 find-first-and-last-position-of-element-in-sorted-array

### python

```python
class Solution(object):
    def searchRange(self, arr, target):
        if not arr:
            return [-1,-1]
        start = self.binary_search(arr, target-0.5)
        if arr[start] != target:
            return [-1, -1]
        arr.append(0)
        end = self.binary_search(arr, target+0.5)-1
        return [start, end]

    def binary_search(self, arr, target):
        start, end = 0, len(arr)-1
        while start < end:
            mid = (start+end)//2
            if target < arr[mid]:
                end = mid
            else:
                start = mid+1
        return start
```

## 15 combination-sum

### python

```python
class Solution(object):
    def combinationSum(self, candidates, target):
        res = []
        candidates.sort()
        self.dfs(candidates, target, 0, [], res)
        return res
    
    def dfs(self, nums, target, index, path, res):
        if target < 0:
            return  # backtracking
        if target == 0:
            res.append(path)
            return 
        for i in xrange(index, len(nums)):
            self.dfs(nums, target-nums[i], i, path+[nums[i]], res)
```



```python
class Solution(object):
    def combinationSum(self, candidates, target):
        candidates.sort()
        stack=[(0,0,[])]
        result=[]
        while stack:
            total,start,res=stack.pop()
            if total ==target:
                result.append(res)
            for i in range(start,len(candidates)):
                t=total+candidates[i]
                if t > target:
                    break
                stack.append((t,i,res+[candidates[i]]))
        return result
```

## 16 combination-sum-ii

### python

```python
class Solution(object):
    def combinationSum2(self, candidates, target):
        res = []
        candidates.sort()
        self.dfs(candidates, target, 0, [], res)
        return res
    
    def dfs(self, nums, target, index, path, res):
        if target < 0:
            return  # backtracking
        if target == 0:
            res.append(path)
            return 
        for i in range(index, len(nums)):
            if i > index and nums[i] == nums[i - 1]:
                continue
            self.dfs(nums, target-nums[i], i+1, path+[nums[i]], res)
```



```python
class Solution(object):
    def combinationSum2(self, candidates, target):
        candidates.sort()
        stack=[(0,0,[])]
        result=[]
        while stack:
            total,start,res=stack.pop()
            if total ==target:
                result.append(res)
            for i in range(start,len(candidates)):
                t=total+candidates[i]
                if t > target:
                    break
                if i >start and candidates[i] == candidates[i-1]:#数组常见去重复的方法，对于重复的数值，我们只让第一个进入循环，后面的就不要再进入循环了
                    continue
                stack.append((t,i+1,res+[candidates[i]]))
        return result
```

## 17 rotate-image

### python

```python
class Solution(object):
    def rotate(self, A):
        A[:]=zip(*A[::-1])
```



```python
class Solution:
    def rotate(self, A):
        A.reverse()
        for i in range(len(A)):
            for j in range(i):
                A[i][j], A[j][i] = A[j][i], A[i][j]
```



# linked

## 206 reverse-linked-list

```python
class Solution(object):
    def reverseList(self, head):
        pre=None
        cur=head
        while cur:
            cur.next,pre,cur=pre,cur,cur.next
        return pre
```



### python

```python
class Solution(object):
    def reverseList(self, head):
        # lterative
        pre=None
        cur=head
        while cur:
            tem=cur.next
            cur.next=pre
            pre=cur
            cur=tem
        return pre
```



```python
class Solution(object):
    def reverseList(self, head):
        return self._serverse(head)
    def _serverse(self,node,pre=None):
        if not node:
            return pre
        n=node.next
        node.next=pre
        return self._serverse(n,node)
```

### go

```go
func reverseList(head *ListNode) *ListNode {
    var pre *ListNode
    return help(head,pre)
}
func help(node *ListNode,pre *ListNode)*ListNode{
    if node ==nil{
        return pre
    }
    n:=node.Next
    node.Next=pre
    return help(n,node)
}
```

```go
func reverseList(head *ListNode) *ListNode {
    var pre *ListNode
	for head!=nil {
        head.Next,head,pre=pre,head.Next,head
		
	}
	return pre
}
```



### C++

```c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode *pre = new ListNode(0), *cur = head;
        pre -> next = head;
        while (cur && cur -> next) {
            ListNode* temp = pre -> next;
            pre -> next = cur -> next;
            cur -> next = cur -> next -> next;
            pre -> next -> next = temp;
        }
        return pre -> next;
    }
};
```



```c++
ListNode* reverseList1(ListNode* head) {
    ListNode *node = NULL;
    while (head) {
        ListNode *nxt = head->next;
        head->next = node;
        node = head;
        head = nxt;
    }
    return node;
}

ListNode* reverseList(ListNode* head) {
    return reverse(head, NULL);
}

ListNode* reverse(ListNode* head, ListNode* node) {
    if (head==nullptr)
        return node;
    ListNode *nxt = head->next;
    head->next = node;
    return reverse(nxt, head);
}
```



## 24 swap-nodes-in-pairs

### python

```python
class Solution(object):
    def swapPairs(self, head):
        pre,pre.next=self,head
        while pre.next and pre.next.next:
            a=pre.next
            b=a.next
            pre.next,b.next,a.next=b,a,b.next
            pre=a
        return self.next
```



```python
 # Recursively    
class Solution(object):
    def swapPairs(self, head):
        if not head or not head.next:
            return head
        result=head.next
        head.next=self.swapPairs(head.next.next)
        result.next=head
        return result
```





### go

```go
func swapPairs(head *ListNode) *ListNode {
    if head == nil || head.Next == nil {
		return head
	}
    first:=head
    second:=head.Next
    first.Next=swapPairs(second.Next)
    second.Next=first
    return second
}
```

```go
func swapPairs(head *ListNode) *ListNode {
    var a ,b *ListNode
    var pp *ListNode=&head
    for  a = *pp && b = a.Next{
        a.Next=b.Next
        b.Next=a
        *pp=b
        pp=&(a.Next)
    }
    return head
}
```



```python
func swapPairs(head *ListNode) *ListNode {
	if head == nil || head.Next == nil {
		return head
	}
	result := head.Next
	head.Next = swapPairs(head.Next.Next)
	result.Next = head
	return result
}
```



### C++

```c++
ListNode* swapPairs(ListNode* head) {
    ListNode **pp = &head, *a, *b;
    while ((a = *pp) && (b = a->next)) {
        a->next = b->next;
        b->next = a;
        *pp = b;
        pp = &(a->next);
    }
    return head;
}
```



```c++
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if(head == NULL)
            return NULL;
        if(head->next == NULL)
            return head;
        
        ListNode* next = head->next;
        head->next = swapPairs(next->next);
        next->next = head;
        
        return next;
    }
};
```



```c++
// recursively
ListNode* swapPairs1(ListNode* head) {
    if (!head || !(head->next))
        return head;
    ListNode *res = head->next;
    head->next = swapPairs(res->next);
    res->next = head;
    return res;
}

// iteratively
ListNode *swapPairs(ListNode *head) {
    ListNode *dummy = new ListNode(0), *node;
    node = dummy;
    dummy->next = head;
    while (head && head->next) {
        ListNode *nxt = head->next;
        head->next = nxt->next;
        nxt->next = head;
        node->next = nxt;
        node = head;
        head = node->next;
    }
    return dummy->next;
}
```



## 03 linked-list-cycle

### python

```python
class Solution(object):
    def hasCycle(self, head):
        fast,slow=head,head
        while fast and fast.next:
            fast=fast.next.next
            slow=slow.next
            if fast==slow:return True
        return False
```

### go

```go
func hasCycle(head *ListNode) bool {
    fast,slow:=head,head
    for fast !=nil && fast.Next !=nil{
        fast=fast.Next.Next
        slow=slow.Next
        if fast == slow{return true}
    }
    return false
}
```



### C++

```c++
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode* slow = head;
		ListNode* fast = head;
		while (fast && fast->next){
			fast = fast->next->next;
			slow = slow->next;
			if (slow == fast)
				return true;
		}
		return false;
    }
}
```



## 04 linked-list-cycle-ii

### python

```python
class Solution(object):
    def detectCycle(self, head):
        fast, slow = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow: break
        else:
            return None
        while head !=fast:
            head=head.next
            fast=fast.next
        return head
```

### go

```go
func detectCycle(head *ListNode) *ListNode {
    if head == nil {
		return nil
	}
	slow, fast := head, head
	for fast.Next != nil && fast.Next.Next != nil {
		fast = fast.Next.Next
		slow = slow.Next
		if slow == fast {
			for head != fast {
				head = head.Next
				fast = fast.Next
			}
			return head
		}
	}
	return nil
}
```

### C++

```c++
ListNode *detectCycle(ListNode *head) {
    if (head == NULL || head->next == NULL)
        return NULL;
    
    ListNode *slow  = head;
    ListNode *fast  = head;
    ListNode *entry = head;
    
    while (fast->next && fast->next->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) {                      // there is a cycle
            while(slow != entry) {               // found the entry location
                slow  = slow->next;
                entry = entry->next;
            }
            return entry;
        }
    }
    return NULL;                                 // there has no cycle
}
```

```c++
class Solution {
    public:
        ListNode *detectCycle(ListNode *head) {
            ListNode* dummy=new ListNode(-1);
            dummy->next=head;
            ListNode *slow=dummy, *fast=dummy;
            bool flag=false;
            while(fast && fast->next){
                slow=slow->next;
                fast=fast->next->next;
                if(fast==slow)  { flag=true; break; }
            }
            if(!flag)   return NULL;
            ListNode* result=dummy;
            while(result != slow){
                result=result->next;
                slow=slow->next;**strong text**
            }
            return result;
        }
    };
```



## 05 reverse-nodes-in-k-group

### python

```python
class Solution(object):
    def reverseKGroup(self, head, k):
        dummy = jump = ListNode(0)
        dummy.next = l = r = head
        while True:
            count = 0
            while r and count < k:   # use r to locate the range
                r = r.next
                count += 1
            if count == k:  # if size k satisfied, reverse the inner linked list
                pre, cur = r, l
                for _ in range(k):
                    cur.next, cur, pre = pre, cur.next, cur  # standard reversing
                jump.next, jump, l = pre, l, r  # connect two k-groups
            else:
                return dummy.next
```



```python
class Solution:
    def reverseList(self, head):
        if not head or not head.next:
            return head
        
        prev, cur, nxt = None, head, head
        while cur:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        return prev 
    
class Solution(object):
    def reverseKGroup(self, head, k):
        count, node = 0, head
        while node and count < k:
            node = node.next
            count += 1
        if count < k: return head
        new_head, prev = self.reverse(head, count)
        head.next = self.reverseKGroup(new_head, k)
        return prev
    
    def reverse(self, head, count):
        prev, cur, nxt = None, head, head
        while count > 0:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
            count -= 1
        return (cur, prev)
```



```python
# Recursively
def reverseKGroup(self, head, k):
    l, node = 0, head
    while node:
        l += 1
        node = node.next
    if k <= 1 or l < k:
        return head
    node, cur = None, head
    for _ in xrange(k):
        nxt = cur.next
        cur.next = node
        node = cur
        cur = nxt
    head.next = self.reverseKGroup(cur, k)
    return node

# Iteratively    
def reverseKGroup(self, head, k):
    if not head or not head.next or k <= 1:
        return head
    cur, l = head, 0
    while cur:
        l += 1
        cur = cur.next
    if k > l:
        return head
    dummy = pre = ListNode(0)
    dummy.next = head
    # totally l//k groups
    for i in xrange(l//k):
        # reverse each group
        node = None
        for j in xrange(k-1):
            nxt = head.next
            head.next = node
            node = head
            head = nxt
        # update nodes and connect nodes
        tmp = head.next
        head.next = node
        pre.next.next = tmp
        tmp1 = pre.next
        pre.next = head
        head = tmp
        pre = tmp1
    return dummy.next
```

### go

```go


```



# stack

## 22 valid-parentheses

### python

```python
 class Solution:
        def isValid(self, s):
            stack=[]
            for i in s:
                if i in ['(','[','{']:
                    stack.append(i)
                else:
                    if not stack or {')':'(',']':'[','}':'{'}[i]!=stack[-1]:
                        return False
                    stack.pop()
            return not stack
```



```python
def isValid(self, s):
    stack, match = [], {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in match:
            if not (stack and stack.pop() == match[ch]):
                return False
        else:
            stack.append(ch)
    return not stack
```

### go

```python
func isValid(s string) bool {
    parentheses := map[rune]rune{')': '(', ']': '[', '}': '{'}
    var stack []rune
    
    for _, char := range s {
        if char == '(' || char == '[' || char == '{' {
            stack = append(stack, char)
        } else if len(stack) > 0 && parentheses[char] == stack[len(stack) - 1] {
            stack = stack[:len(stack) - 1]
        } else {
            return false
        }
    }
    return len(stack) == 0
}
```



## 155 min-stack

### python

```python
class MinStack:

def __init__(self):
    self.q = []

# @param x, an integer
# @return an integer
def push(self, x):
    curMin = self.getMin()
    if curMin == None or x < curMin:
        curMin = x
    self.q.append((x, curMin));

# @return nothing
def pop(self):
    self.q.pop()


# @return an integer
def top(self):
    if len(self.q) == 0:
        return None
    else:
        return self.q[len(self.q) - 1][0]


# @return an integer
def getMin(self):
    if len(self.q) == 0:
        return None
    else:
        return self.q[len(self.q) - 1][1]
```

## 84 largest-rectangle-in-histogram

### python

```python
def largestRectangleArea(self, height):
    height.append(0) # very important!!
    stack = [-1]
    ans = 0
    for i in xrange(len(height)):
        while height[i] < height[stack[-1]]:
            h = height[stack.pop()]
            w = i - stack[-1] - 1
            ans = max(ans, h * w)
        stack.append(i)
    height.pop()
    return ans
```



```python
class Solution(object):
    def largestRectangleArea(self, heights):
        n = len(heights)
        left = [1]*n
        right = [1]*n
        
        for i in range(1,n):
            j = i - 1
            while j>=0 and heights[j] >= heights[i]:
                j -= left[j]
            left[i] = i - j
             
        for i in range(n-2,-1,-1):
            j = i + 1
            while j < len(heights) and heights[i] <= heights[j]:
                j += right[j] 
            right[i] = j - i
            
        res = 0
        for i in range(n):
            res = max(res,heights[i]*(left[i]+right[i]-1))
                
        return res
```

### go

```go
func largestRectangleArea(heights []int) int {

	if len(heights) == 0 {
		return 0
	}
	if len(heights) == 1 {
		return heights[0]
	}
	size := len(heights)
	stack := make([]int, 0, size)

	heights = append(heights, 0)
	stack = append(stack, -1) // 添加哨兵
	i := 0
	res := 0

	for i < size+1 && len(stack) != 0 {
		cur := heights[i]
		topindex := stack[len(stack)-1]
		if topindex < 0||cur >= heights[topindex] { // 构造递增栈
			// 入栈
			stack = append(stack, i)
			i++
			continue
		}

		stack = stack[:len(stack)-1]
		// 栈中弹出高的元素之后的值
		begin := stack[len(stack)-1]

		area := (i - begin - 1) * heights[topindex]
		res = max(res, area)
	}
	return res
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
```



```go
func largestRectangleArea(heights []int) int {
    size:=len(heights)
    heights=append(heights,0)
    stack:=make([]int,0,size)

    stack=append(stack,-1)  //哨兵
    i,res:=0,0

    for i<size+1 && len(stack) !=0{
        cur :=heights[i]
        toindex:=stack[len(stack)-1]
        if toindex <0 || cur >heights[toindex]{
            stack=append(stack,i)
            i++
            continue
        }
        stack=stack[:len(stack)-1]
        begin:=stack[len(stack)-1]
        if (i-begin-1)*heights[toindex] > res{
            res=(i-begin-1)*heights[toindex]
        }
    }
    return res
}
```



### C++

```c++
class Solution {
    public:
        int largestRectangleArea(vector<int> &height) {
            
            int ret = 0;
            height.push_back(0);
            vector<int> index;
            
            for(int i = 0; i < height.size(); i++)
            {
                while(index.size() > 0 && height[index.back()] >= height[i])
                {
                    int h = height[index.back()];
                    index.pop_back();
                    
                    int sidx = index.size() > 0 ? index.back() : -1;
                    if(h * (i-sidx-1) > ret)
                        ret = h * (i-sidx-1);
                }
                index.push_back(i);
            }
            
            return ret;
        }
    };
```



```c++
class Solution {
public:
    int largestRectangleArea(vector<int>& heights) {
        int ret=0,i=0;
        heights.push_back(0);
        stack<int> stk;
        int size_h=heights.size();
        while (i<size_h){
            if (stk.empty() || heights[i] >heights[stk.top()]) stk.push(i++);
            else{
                int h=stk.top();
                stk.pop();
                ret=max(ret,heights[h]*(stk.empty()?i:i-stk.top()-1));
            }
        }
        return ret;
    }
};
```



## 239 sliding-window-maximum

### python

```python
class Solution(object):
    def maxSlidingWindow(self, nums, k): #暴力
        n=len(nums)
        if n*k==0:
            return []
        return [max(nums[i:i+k])for i in range(n-k+1)]
```



```python
lass Solution(object):
    def maxSlidingWindow(self, nums, k):
        if not nums:
            return
        ans = []
        queue = []
        for i, v in enumerate(nums):
            if i>=k and queue[0] <= i - k:
                queue.pop(0)
            while queue and nums[queue[-1]] < v:
                queue.pop()
            queue.append(i)
            if i + 1 >= k:
                ans.append(nums[queue[0]])
        return ans
```



```python
def maxSlidingWindow(self, nums, k):
    d = collections.deque()
    out = []
    for i, n in enumerate(nums):
        while d and nums[d[-1]] < n:
            d.pop()
        d += i,
        if d[0] == i - k:
            d.popleft()
        if i >= k - 1:
            out += nums[d[0]],
    return out
```



```python
import heapq as h
class Solution(object):
    def get_next_max(self, heap, start):
        while True:
            x,idx = h.heappop(heap)
            if idx >= start:
                return x*-1, idx
    
    def maxSlidingWindow(self, nums, k):
        if k == 0:
            return []
        heap = []
        for i in range(k):
            h.heappush(heap, (nums[i]*-1, i))
        result, start, end = [], 0, k-1
        while end < len(nums):
            x, idx = self.get_next_max(heap, start)
            result.append(x)
            h.heappush(heap, (x*-1, idx)) 
            start, end = start + 1, end + 1
            if end < len(nums):
                h.heappush(heap, (nums[end]*-1, end))
        return result
```



### go

```go
func maxSlidingWindow(nums []int, k int) []int {
	result := make([]int, 0, len(nums)-k)
	window := make([]int, 0)

	for i := 0; i < len(nums); i++ {
		if len(window) > 0 && window[0] <= i-k {
			window = window[1:]
		}
		for len(window) > 0 && nums[window[len(window)-1]] < nums[i] {
			window = window[:len(window)-1]
		}
		window = append(window, i)
		if i >= k-1 {
			result = append(result, nums[window[0]])
		}
	}
	return result
}
```

### C++

```c++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> dq;
        vector<int> ans;
        for (int i=0; i<nums.size(); i++) {
            if (!dq.empty() && dq.front() == i-k) dq.pop_front();
            while (!dq.empty() && nums[dq.back()] < nums[i])
                dq.pop_back();
            dq.push_back(i);
            if (i>=k-1) ans.push_back(nums[dq.front()]);
        }
        return ans;
    }
};
```

```c++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> buffer;
        vector<int> res;

        for(auto i=0; i<nums.size();++i)
        {
            while(!buffer.empty() && nums[i]>=nums[buffer.back()]) buffer.pop_back();
            buffer.push_back(i);

            if(i>=k-1) res.push_back(nums[buffer.front()]);
            if(buffer.front()<= i-k + 1) buffer.pop_front();
        }
        return res;
    }
};
```



# queue

## 703 kth-largest-element-in-a-stream

### python

```python
class KthLargest(object):
    def __init__(self, k, nums):
        self.pool = nums
        self.k = k
        heapq.heapify(self.pool)
        while len(self.pool) > k:
            heapq.heappop(self.pool)
    def add(self, val):
        if len(self.pool) < self.k:
            heapq.heappush(self.pool, val)
        elif val > self.pool[0]:
            heapq.heapreplace(self.pool, val)
        return self.pool[0]
```

### go

```python
// copied from golang doc
// mininum setup of integer min heap
type IntHeap []int

func (h IntHeap) Len() int           { return len(h) }
func (h IntHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h IntHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *IntHeap) Push(x interface{}) {
	// Push and Pop use pointer receivers because they modify the slice's length,
	// not just its contents.
	*h = append(*h, x.(int))
}

func (h *IntHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

// real thing starts here
type KthLargest struct {
	Nums *IntHeap
	K    int
}

func Constructor(k int, nums []int) KthLargest {
	h := &IntHeap{}
	heap.Init(h)
	// push all elements to the heap
	for i := 0; i < len(nums); i++ {
		heap.Push(h, nums[i])
	}
	// remove the smaller elements from the heap such that only the k largest elements are in the heap
	for len(*h) > k {
		heap.Pop(h)
	}
	return KthLargest{h, k}
}

func (this *KthLargest) Add(val int) int {
	if len(*this.Nums) < this.K {
		heap.Push(this.Nums, val)
	} else if val > (*this.Nums)[0] {
		heap.Push(this.Nums, val)
		heap.Pop(this.Nums)
	}
	return (*this.Nums)[0]
}
```



## 239 sliding-window-maximum

### python

```python
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        if not nums:
            return
        ans = []
        queue = []
        for i, v in enumerate(nums):
            if i>=k and queue[0] <= i - k:
                queue.pop(0)
            while queue and nums[queue[-1]] < v:
                queue.pop()
            queue.append(i)
            if i + 1 >= k:
                ans.append(nums[queue[0]])
        return ans
```



# 哈希表

## 242 valid-anagram

### python

```python
def isAnagram1(self, s, t):
    dic1, dic2 = {}, {}
    for item in s:
        dic1[item] = dic1.get(item, 0) + 1
    for item in t:
        dic2[item] = dic2.get(item, 0) + 1
    return dic1 == dic2
    
def isAnagram2(self, s, t):
    dic1, dic2 = [0]*26, [0]*26
    for item in s:
        dic1[ord(item)-ord('a')] += 1
    for item in t:
        dic2[ord(item)-ord('a')] += 1
    return dic1 == dic2
    
def isAnagram3(self, s, t):
    return sorted(s) == sorted(t)
```

### go

```go
func isAnagram(s string, t string) bool {
    var a [26]int
    var b [26]int
    for _,v:=range s{
        a[v-'a']++
    }
    for _,v:=range t{
        b[v-'a']++
    }
    return a ==b
}
```

```go
func isAnagram(s string, t string) bool {
   map_dic1:=make(map[rune]int)
   if len(s) !=len(t){return false}
   for _,v:=range s{
       if _,ok:=map_dic1[v];ok{
           map_dic1[v]++
       }else{
           map_dic1[v]=1
       }
   }
    
   for _,v:=range t{
       if _,ok:=map_dic1[v];ok && map_dic1[v] >0{
           map_dic1[v]--
           if map_dic1[v]==0{delete(map_dic1,v)}
       }else{
           return false
       }
   }
    if len(map_dic1)==0{return true}
    return false
}
```

### C++

```c++
class Solution {
public:
    bool isAnagram(string s, string t) {
        if(s.size() != t.size())
            return false;
            
        int hash[26]={0};       //构建哈希数组
        for(auto n:s)
            hash[n-'a']++;
        for(auto n:t)
            hash[n-'a']--;
        for(int i=0;i<26;i++)
            if(hash[i]!=0)   return false;          //如果两数组不完全相等
        return true;
    }
};

class Solution {
public:
    bool isAnagram(string s, string t) {
        if(s.size()!=t.size())
        return false;
        sort(s.begin(),s.end());
        sort(t.begin(),t.end());

        return s==t;
		
    }
};

class Solution {
public:
    bool isAnagram(string s, string t) {
        if (s.size() != t.size()) {
            return false;
        }
        unordered_map<char, int> umap;
        for (int i = 0; i < s.size(); ++i) {
            ++umap[s[i]];
            --umap[t[i]];
        }
        for (auto c : umap) {
            if (c.second != 0) {
                return false;
            }
        }
        return true;
    }
};

class Solution {
public:
    bool isAnagram(string s, string t) {
        if (s.size() !=t.size())return false;
        vector<int> hash(26);
        for (int i=0;i<t.size();i++){
            ++hash[t[i]-'a'];
            --hash[s[i]-'a'];
        }
        for (auto c:hash)
        if (c !=0) return false;
        return true;
    }
};
```



## 49 group-anagrams

### python

```python
def groupAnagrams(self, strs):
    d = {}
    for w in sorted(strs):
        key = tuple(sorted(w))
        d[key] = d.get(key, []) + [w]
    return d.values()
```

```python
class Solution(object):
    def groupAnagrams(self, strs):
        dic={}
        for str in strs:
            key="".join(sorted(str))
            if key in dic:
                dic[key].append(str)
            else:
                dic[key]=[str]
        return list(dic.values())
```

### go

```go
func groupAnagrams(strs []string) [][]string {
    m := make(map[string][]string, 0)
    result := make([][]string, 0)
	for _, val := range strs {
		strArr := strings.Split(val, "")
		sort.Strings(strArr)
		sortedString := strings.Join(strArr, "")
		m[sortedString] = append(m[sortedString], val)
	}

	for _, value := range m {
		result = append(result, value)
	}
	return result
}
```

```go
func groupAnagrams(words []string) [][]string {
	cache := make(map[[26]byte]int)
	result := make([][]string, 0)
	for i := range words {
		list := [26]byte{}
		for j := range words[i] {
			list[words[i][j]-'a']++
		}
		if idx, ok := cache[list]; ok {
			result[idx] = append(result[idx], words[i])
		} else {
			result = append(result, []string{words[i]})
			cache[list] = len(result) - 1
		}
	}
	return result
}
```

### C++

```c++
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string,vector<string>> mp;
        for (string s:strs){
            string t=s;
            sort(t.begin(),t.end());
            mp[t].push_back(s);
        }
        vector<vector<string>> res;
        for (auto p:mp){
            res.push_back(p.second);
        }
        //迭代器
        vector<vector<string>> res;
        for (auto x=mp.begin();x!=mp.end();x++){
            vector<string> tem=x->second;
            res.push_back(tem);
        }
        return res 
    }
};
```



# tree

## 94  binary-tree-inorder-traversal

### python

```python
# recursively
def inorderTraversal1(self, root):
    res = []
    self.helper(root, res)
    return res
    
def helper(self, root, res):
    if root:
        self.helper(root.left, res)
        res.append(root.val)
        self.helper(root.right, res)
 
# iteratively       
def inorderTraversal(self, root):
    res, stack = [], []
    while True:
        while root:
            stack.append(root)
            root = root.left
        if not stack:
            return res
        node = stack.pop()
        res.append(node.val)
        root = node.right
```



```python
class Solution:
    def inorderTraversal(self, root):
        result, stack = [], [(root, False)]
        while stack:
            cur, visited = stack.pop()
            if cur:
                if visited:
                    result.append(cur.val)
                else:
                    stack.append((cur.right, False))
                    stack.append((cur, True))
                    stack.append((cur.left, False))
        return result
```



### go



```go
var res []int

func inorderTraversal(root *TreeNode) []int {
	res = make([]int, 0)
	dfs(root)
	return res
}

func dfs(root *TreeNode) {
	if root != nil {
		dfs(root.Left)
		res = append(res, root.Val)
		dfs(root.Right)
	}
}
```



```go
func inorderTraversal(root *TreeNode) []int {
    var res []int
	var stack []*TreeNode
	for root !=nil || len(stack)>0{
		for root !=nil{
			stack=append(stack,root)
			root=root.Left
		}
		pre:=len(stack)-1
		res=append(res,stack[pre].Val)
		root=stack[pre].Right
		stack=stack[:pre]
	}
	return res
}
```



```go
func inorderTraversal(root *TreeNode) []int {
    ret := make([]int, 0)
	if root == nil {
		return ret
	}
	stack := list.New()
	for root != nil || stack.Len() != 0 {
		for root != nil {
			fmt.Println(root.Val)
			stack.PushBack(root)
			root = root.Left
		}
		root = stack.Back().Value.(*TreeNode)
		ret = append(ret, root.Val)
		stack.Remove(stack.Back())
		root = root.Right
	}
	return ret
}
```

### C++

```c++
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
         vector<int> res;
         dfs(root,res);
         return res;
    }
private:
    void dfs(TreeNode* root,vector<int>& res){
        if (root) {
        dfs(root->left,res);
        res.push_back(root->val);
        dfs(root->right,res);
        }
    }
};
```

```c++
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> stk;
        while (true){
            while (root){
                stk.push(root);
                root=root->left;
            }
            if (stk.size()==0) return res;
            TreeNode* node=stk.top();
            stk.pop();
            res.push_back(node->val);
            root=node->right;
        }
    }
};
```



## 144 binary-tree-preorder-traversal

### python

```python
class Solution(object):
    def preorderTraversal(self, root):
        if root is None:
            return []
        
        stack, output = [root, ], []
        
        while stack:
            root = stack.pop()
            if root is not None:
                output.append(root.val)
                if root.right is not None:
                    stack.append(root.right)
                if root.left is not None:
                    stack.append(root.left)
        return output

```



```python
# recursively
def preorderTraversal1(self, root):
    res = []
    self.dfs(root, res)
    return res
    
def dfs(self, root, res):
    if root:
        res.append(root.val)
        self.dfs(root.left, res)
        self.dfs(root.right, res)

# iteratively
def preorderTraversal(self, root):
    stack, res = [root], []
    while stack:
        node = stack.pop()
        if node:
            res.append(node.val)
            stack.append(node.right)
            stack.append(node.left)
    return res
```



### go

```go
var res []int
func preorderTraversal(root *TreeNode) []int {
    res=make([]int,0)
    dfs(root)
    return res
}
func dfs(root *TreeNode){
    if root !=nil{
        res=append(res,root.Val)
        dfs(root.Left)
        dfs(root.Right)
    }
}
```



```go
func preorderTraversal(root *TreeNode) []int {
	if root == nil {
		return nil
	}
	var stack []*TreeNode
	stack = append(stack, root)

	var ret []int
	for len(stack) > 0 {
		p := stack[len(stack)-1]
		stack = stack[0 : len(stack)-1]
		ret = append(ret, p.Val)
		if p.Right != nil {
			stack = append(stack, p.Right)
		}
		if p.Left != nil {
			stack = append(stack, p.Left)
		}
	}

	return ret
}
```



```go
func preorderTraversal(root *TreeNode) []int {
    if root ==nil{return nil}
    var res []int
    var stack []*TreeNode
    stack=append(stack,root)
    for len(stack) >0{
        node:=stack[len(stack)-1]
        stack=stack[:len(stack)-1]
        if node !=nil{
            res=append(res,node.Val)
            stack=append(stack,node.Right)
            stack=append(stack,node.Left)
        }
    }
    return res
}
}
```



### C++

```c++
class Solution {
public:
    vector<int> res;
    vector<int> preorderTraversal(TreeNode* root) {
        dfs(root);
        return res;
    }
private:
    void dfs(TreeNode* root){
        if (root){
            res.push_back(root->val);
            dfs(root->left);
            dfs(root->right);
        }
    }
};
```



```c++
class Solution {
public:
    vector<int> res;
    vector<int> preorderTraversal(TreeNode* root) {
        stack<TreeNode*> stk;
        TreeNode* node = root;
        stk.push(root);
        while (stk.size() >0){
            node=stk.top();
            stk.pop();
            if (node){
                res.push_back(node->val);
                stk.push(node->right);
                stk.push(node->left);
            }
        }
        return res;
    }
};
```



## 590 n-ary-tree-postorder-traversal

### python

```python
class Solution(object):
    def postorder(self, root):
        res=[]
        self.dfs(root,res)
        return res
    def dfs(self,root,res):
        if root:
            for node in root.children:
                self.dfs(node,res)
            res.append(root.val
```



```python
class Solution(object):
    def postorder(self, root):
        if root is None:
            return []
        
        stack, output = [root, ], []
        while stack:
            root = stack.pop()
            if root is not None:
                output.append(root.val)
            for c in root.children:
                stack.append(c)
        return output[::-1]
```



```python
class Solution(object):
    def postorder(self, root):
        return [] if not root else [j for i in root.children for j in self.postorder(i)] + [root.val]
```



```python
def postorder(self, root):
        """
        :type root: Node
        :rtype: List[int]
        """
        res = []
        if root == None: return res

        def recursion(root, res):
            for child in root.children:
                recursion(child, res)
            res.append(root.val)

        recursion(root, res)
        return res
```



```python
def postorder(self, root):
        res = []
        if root == None: return res

        stack = [root]
        while stack:
            curr = stack.pop()
            res.append(curr.val)
            stack.extend(curr.children)

        return res[::-1]
```

### go

```go
var res []int
func postorder(root *Node) []int {
    res=make([]int,0)
    dfs(root)
    return res
}
func dfs(root *Node){
    if root !=nil{
        for _,v:=range root.Children{
            dfs(v)
        }
        res=append(res,root.Val)
    }
}
```



```go
func postorder(root *Node) []int {
    res := []int{}
    if root == nil {
        return res
    }
    
    stack := list.New()
    stack.PushFront(root)
    for stack.Len() > 0 {
        curr := stack.Remove(stack.Front()).(*Node)
        if curr != nil {
            res = append(res, curr.Val)        
        }
    
        for i := 0; i < len(curr.Children) ; i++ {
            stack.PushFront(curr.Children[i])
        }
    }
    
    reverse(res)
    
    return res
}

func reverse (s []int) {
    for i, j := 0, len(s)-1; i < j; i, j = i+1, j-1 {
        s[i], s[j] = s[j], s[i]
    }
}
```



# recursion

## 递归模板

```python
def recursion(level, param1, param2, ...): 
    # recursion terminator 
    if level > MAX_LEVEL: 
	   process_result 
	   return 

    # process logic in current level 
    process(level, data...) 

    # drill down 
    self.recursion(level + 1, p1, ...) 

    # reverse the current level status if needed
```





## 22 generate-parentheses

### python



```python
class Solution(object):
    def generateParenthesis(self, n):
        if not n:
            return []
        left,right,ans=n,n,[]
        self.recursion(left,right,ans,"")
        return ans
    def recursion(self,left,right,ans,s):
        if right<left:
            return 
        #recursion terminato
        if not left and not right:
            ans.append(s)
            return
        # process logic in current level
        # drill down
        if left:
            self.recursion(left-1,right,ans,s+"(")
        if right:
            self.recursion(left , right-1,ans, s + ")")
        # reverse the current level status if needed
```



```python
def generateParenthesis(self, n):
    res = []
    self.dfs(n, n, "", res)
    return res
        
def dfs(self, leftRemain, rightRemain, path, res):
    if leftRemain > rightRemain or leftRemain < 0 or rightRemain < 0:
        return  # backtracking
    if leftRemain == 0 and rightRemain == 0:
        res.append(path)
        return 
    self.dfs(leftRemain-1, rightRemain, path+"(", res)
    self.dfs(leftRemain, rightRemain-1, path+")", res）
```

**dp解法**

```python
class Solution(object):
    def generateParenthesis(self, n):
        dp=[[] for i in range(n+1)]
        dp[0].append("")
        for i in range(n+1):
            for j in range(i):
                dp[i]+=["("+x+")"+y for x in dp[j] for y in dp[i-j-1]]
        return dp[n]
```



### go

```go
func generateParenthesis(n int) []string {
    res:=make([]string,0)
    dfs(n,n,"",&res)
    return res
}
func dfs(left,right int,path string,res *[]string){
    if right <left || left <0 || right <0{return}
    if left==0 && right == 0{
        *res=append(*res,path)
        return
    }
    dfs(left-1,right,path+"(",res)
    dfs(left,right-1,path+")",res)
}
```



```go
func generateParenthesis(n int) []string {
	m:=map[string]bool{}
	res:=make([]string,0)
	dfs(n,n,"",m)
	for k :=range m{
		res = append(res, k)
	}
	return res
}
func dfs(left,right int,path string,cache map[string]bool)  {
	if left> right || left<0 || right<0{
		return
	}
	if left==0 && right==0{
		cache[path]=true
		return
	}
	dfs(left-1,right,path+"(",cache)
	dfs(left,right-1,path+")",cache)
}
```

### C++

```c++
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        vector<string> res;
        dfs(n,n,"",res);
        return res;
    }
    void dfs(int left,int right,string path,vector<string>& res){
        if (right < left || right < 0 || left < 0) return;
        if (left == 0 && right == 0){
            res.push_back(path);
            return;
        }
        dfs(left-1,right,path+"(",res);
        dfs(left,right-1,path+")",res);
    }
};
```



## 226 invert-binary-tree

### python

```python
class Solution(object):
    def invertTree(self, root):
        # one silve
        if not root:
            return None
        root.left,root.right=root.right,root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
        
        #two selve
        if not root:
            return root
        res=[root]
        while res:
            current=res.pop()
            if current:
                current.left, current.right = current.right, current.left
                res+=current.left,current.right
        return root
```



```python
def invertTree(self, root):
    if root:
        invert = self.invertTree
        root.left, root.right = invert(root.right), invert(root.left)
        return root

def invertTree(self, root):
    stack = [root]
    while stack:
        node = stack.pop()
        if node:
            node.left, node.right = node.right, node.left
            stack += node.left, node.right
    return root
```



```python
# recursively
def invertTree1(self, root):
    if root:
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root
        
# BFS
def invertTree2(self, root):
    queue = collections.deque([(root)])
    while queue:
        node = queue.popleft()
        if node:
            node.left, node.right = node.right, node.left
            queue.append(node.left)
            queue.append(node.right)
    return root
    
# DFS
def invertTree(self, root):
    stack = [root]
    while stack:
        node = stack.pop()
        if node:
            node.left, node.right = node.right, node.left
            stack.extend([node.right, node.left])
    return root
```

### go

```go
func invertTree(root *TreeNode) *TreeNode {
	if root ==nil{
	return nil
	}
	root.Left,root.Right=root.Right,root.Left
	invertTree(root.Left)
	invertTree(root.Right)
	return root
}
```



```go
func invertTree(root *TreeNode) *TreeNode {
	if root ==nil{
	return nil
	}
    res:= []*TreeNode{}
	res = append(res, root)
	for len(res) >0{
		pre:=len(res)-1
        current:=res[pre]
		res=res[:pre]
		current.Left, current.Right = current.Right, current.Left
		if current.Left !=nil{
			res = append(res, current.Left)
		}
		if current.Right !=nil{
			res = append(res, current.Right)
		}
	}
	return root
}
```

### C++

```c++
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (root ==NULL) return NULL;
        swap(root->left,root->right);
        invertTree(root->left);
        invertTree(root->right);
        return root;
    }
};
```



## 98 validate-binary-search-tree

### python

```python
class Solution(object):
    def isValidBST(self, root):
        inorder=self.inorder(root)
        return inorder==list(sorted(set(inorder)))
    def inorder(self,root):
        if not root:
            return []
        return self.inorder(root.left)+[root.val]+self.inorder(root.right)
    #优化代码
    def isValidBST(self, root):
        self.pre=None
        return self.dfs(root)
    def dfs(self,root):
        if not root:
            return True

        if not self.dfs(root.left):
            return False
        if self.pre and self.pre.val >=root.val:
            return False
        self.pre=root
        return self.dfs(root.right)
```



```python
class Solution(object):
    def isValidBST(self, root, lessThan = float('inf'), largerThan = float('-inf')):
        if not root:
            return True
        if root.val <= largerThan or root.val >= lessThan:
            return False
        return self.isValidBST(root.left, min(lessThan, root.val), largerThan) and \
               self.isValidBST(root.right, lessThan, max(root.val, largerThan))
```





### go

```go
func isValidBST(root *TreeNode) bool {
    res:=make([]int,0)
    dfs(root,&res)
    for i:=1;i<len(res);i++{
        if res[i-1]>=res[i]{return false}
    }
    return true
}
func dfs(root *TreeNode,res *[]int) {
	if root != nil {
		dfs(root.Left,res)
		*res = append(*res, root.Val)
		dfs(root.Right,res)
	}
}
//优化? 有b
func isValidBST(root *TreeNode) bool {
    var pre *TreeNode
    return dfs(root,pre)
}
func dfs(root *TreeNode,pre *TreeNode)bool{
    if root ==nil{return true}
    if !dfs(root.Left,pre) {return false}
    if (pre !=nil && pre.Val >= root.Val){return false}
    pre=root
    return dfs(root.Right,pre)
}
```

```go
func isValidBST(root *TreeNode) bool {
    return RecValidate(root,nil,nil)
}
func RecValidate(n,min,max *TreeNode) bool{
    if n==nil{return true}
    if min !=nil && n.Val <=min.Val{return false}
    if max !=nil && n.Val >=max.Val{return false}
    return RecValidate(n.Left,min,n) && RecValidate(n.Right,n,max)
}
```



### C++

```c++
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        vector<int> res;
        dfs(root,res);
        for (int i=1;i<res.size();i++)
        if (res[i-1] >=res[i]) return false;
        return true;
    }
    void dfs(TreeNode* root,vector<int>& res){
        if (root){
            dfs(root->left,res);
            res.push_back(root->val);
            dfs(root->right,res);
        }
    }
};

//优化
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        TreeNode* pre=NULL;
        return validate(root,pre);
    }
    bool validate(TreeNode* node,TreeNode* &pre){
        if (node ==NULL) return true;
        if (!validate(node->left,pre)) return false;
        if (pre && pre->val >=node->val) return false;
        pre=node;
        return validate(node->right,pre);
    }
};
```

```c++
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        return dfs(root,NULL,NULL);
    }
    bool dfs(TreeNode* root,TreeNode* low,TreeNode* high){
        if (root ==NULL) return true;
        if (low && root->val<=low->val || high && root->val>=high->val) return false;
        return dfs(root->left,low,root) && dfs(root->right,root,high);
    }
};
```



## 104 maximum-depth-of-binary-tree

### python

```python
class Solution(object):
    def maxDepth(self, root):
        #递归
        if not root:
            return 0
        else:
            left_hight=self.maxDepth(root.left)
            right_hight=self.maxDepth(root.right)
            return max(left_hight,right_hight)+1
        
        #迭代
        stack=[]
        if root is not None:
            stack.append((1,root))
        depth=0
        while stack:
            current_depth,root=stack.pop()
            if root is not None:
                depth=max(depth,current_depth)
                stack.append((current_depth+1,root.left))
                stack.append((current_depth+1,root.right))
        return depth
```



```python
def maxDepth(self, root):
    return 1 + max(map(self.maxDepth, (root.left, root.right))) if root else 0
```



```python
class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        depth = 0
        level = [root] if root else []
        while level:
            depth += 1
            queue = []
            for el in level:
                if el.left:
                    queue.append(el.left)
                if el.right:
                    queue.append(el.right)
            level = queue
            
        return depth
```



```python
# BFS + deque   
def maxDepth(self, root):
    if not root:
        return 0
    from collections import deque
    queue = deque([(root, 1)])
    while queue:
        curr, val = queue.popleft()
        if not curr.left and not curr.right and not queue:
            return val
        if curr.left:
            queue.append((curr.left, val+1))
        if curr.right:
            queue.append((curr.right, val+1))
```



### go



```go
func maxDepth(root *TreeNode) int {
    if root != nil {
        leftDepth := maxDepth(root.Left)
        rightDepth := maxDepth(root.Right)
        if leftDepth > rightDepth {
            return 1+leftDepth
        }
        return 1+rightDepth
    }
    return 0
}
```

## 111 minimum-depth-of-binary-tree

### python

```python
def minDepth(self, root):
    if not root: return 0
    d = map(self.minDepth, (root.left, root.right))
    return 1 + (min(d) or max(d))

def minDepth(self, root):
    if not root: return 0
    d, D = sorted(map(self.minDepth, (root.left, root.right)))
    return 1 + (d or D)
```



```python
class Solution(object):
    def minDepth(self, root):
        if not root:
            return 0
        if not root.left:return 1+self.minDepth(root.right)
        if not root.right:return 1+self.minDepth(root.left)
        left=self.minDepth(root.left)
        right=self.minDepth(root.right)
        result=1+min(left,right)
        return result
 
# BFS   
def minDepth(self, root):
    if not root:
        return 0
    queue = collections.deque([(root, 1)])
    while queue:
        node, level = queue.popleft()
        if node:
            if not node.left and not node.right:
                return level
            else:
                queue.append((node.left, level+1))
                queue.append((node.right, level+1))
```

### go

```go
func minDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	left, right := minDepth(root.Left), minDepth(root.Right)
	if left == 0 || right == 0 {
		return 1 + left + right
	}
	return 1 + int(math.Min(float64(left), float64(right)))
}
```



```go'
func minDepth(root *TreeNode) int {
    if root ==nil{return 0}
    depth:=1
    node:=[]*TreeNode{root}
    for{
        newNode:=[]*TreeNode{}
        for _,n:=range node{
            if n.Left==nil && n.Right==nil{return depth}
            if n.Left !=nil{newNode=append(newNode,n.Left)}
            if n.Right !=nil{newNode=append(newNode,n.Right)}
        }
        depth++
        node=newNode
    }
    return depth
}
```

# 分治 回溯

```python
def divide_conquer(problem, param1, param2, ...): 
  # recursion terminator 
  if problem is None: 
	print_result 
	return 

  # prepare data 
  data = prepare_data(problem) 
  subproblems = split_problem(problem, data) 

  # conquer subproblems 
  subresult1 = self.divide_conquer(subproblems[0], p1, ...) 
  subresult2 = self.divide_conquer(subproblems[1], p1, ...) 
  subresult3 = self.divide_conquer(subproblems[2], p1, ...) 
  …

  # process and generate the final result 
  result = process_result(subresult1, subresult2, subresult3, …)
	
  # revert the current level states
```



## 50 powx-n

### python

```python
class Solution(object):
    def myPow(self, x, n):
        def func(x,n):
            if n==0:
                return 1
            if x == 0:
                return 0
            tem=func(x,n>>1)
            if n & 1:
                return tem*tem*x
            else:
                return tem*tem
        if n >=0:
            res=func(x,n)
        else:
            res=1/func(x,-n)
        return res
```





```python
class Solution:
    def myPow(self, x, n):
        if not n:
            return 1
        if n < 0:
            return 1 / self.myPow(x, -n)
        if n % 2:
            return x * self.myPow(x, n-1)
        return self.myPow(x*x, n/2)

class Solution:
    def myPow(self, x, n):
        if n < 0:
            x = 1 / x
            n = -n
        pow = 1
        while n:
            if n & 1:
                pow *= x
            x *= x
            n >>= 1
        return pow
```



```python
class Solution:
    def myPow(self, a, b):
        if b == 0: return 1
        if b < 0: return 1.0 / self.myPow(a, -b)
        half = self.myPow(a, b // 2)
        if b % 2 == 0:
            return half * half
        else:
            return half * half * a
```



### go

```go
func myPow(x float64, n int) float64 {
    if n ==0{return 1}
    if n <0{return 1/myPow(x,-n)}
    half:=myPow(x,n>>1)
    if n % 2 ==0{
        return half*half
    }
    return half*half*x
}
```

### C++

```c++
class Solution {
public:
    double myPow(double x, int n) {
        int half;
        if (n ==0) return 1;
        if (n <0) return 1/myPow(x,-n);
        half=myPow(x,n>>1);
        if (n & 1) return half*half*x;
        else return half*half;
    }
};
```



## 78 subsets

### python



```python
class Solution(object):
    def subsets(self, nums):
        res=[[]]
        for i in nums:
            res=res+[[i]+num for num in res]
        return res
```



```python
class Solution(object):
    def subsets(self, nums):
        res = []
        self.dfs(sorted(nums), 0, [], res)
        return res
        
    def dfs(self, nums, index, path, res):
        res.append(path)
        for i in xrange(index, len(nums)):
            self.dfs(nums, i+1, path+[nums[i]], res)
        
# Bit Manipulation    
def subsets2(self, nums):
    res = []
    nums.sort()
    for i in xrange(1<<len(nums)):
        tmp = []
        for j in xrange(len(nums)):
            if i & 1 << j:  # if i >> j & 1:
                tmp.append(nums[j])
        res.append(tmp)
    return res
    
```



### go

```go
func subsets(nums []int) [][]int {
    return helper(nums, 0, [][]int{{}})
}

func helper(nums []int, i int, sets [][]int) [][]int {
    if i == len(nums) {
        return sets
    }
    for _, s := range sets {
        sets = append(sets, append([]int{nums[i]}, s...))
    }
    return helper(nums, i+1, sets)
}
```





```go
func subsets(nums []int) [][]int{
    var results [][]int
    var result []int
    qst(nums)
    //sort.Ints(nums)
    results=append(results,[]int{})
    find(nums,&result,&results)
    return results
}

func find(nums []int,result *[]int,results *[][]int){
    for i:=0;i<len(nums);i++{
        *result=append(*result,nums[i])
        var cc []int
        for _,v:=range *result{
            cc=append(cc,v)
        }
        *results=append(*results,cc)
        find(nums[i+1:],result,results)
        *result=(*result)[:len(*result)-1]
    }
}

func qst(nums []int){
    if len(nums)==1||len(nums)==0{
        return
    }
    mid:=nums[0]
    i,j:=0,len(nums)-1
    for i<j{
        for i<j&&mid<=nums[j]{
            j--
        }
        nums[i],nums[j]=nums[j],nums[i]
        for i<j&&mid>=nums[i]{
            i++
        }
        nums[i],nums[j]=nums[j],nums[i]
    }
    qst(nums[:i])
    qst(nums[i+1:])
}
```

### C++

```c++
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int> tem;
        dfs(nums,0,res,tem);
        return res;
    }
    void dfs(vector<int>& nums,int i,vector<vector<int>>& res,vector<int>& tem){
        res.push_back(tem);
        for (int j=i;j<nums.size();j++){
            tem.push_back(nums[j]);
            dfs(nums,j+1,res,tem);
            tem.pop_back();
        }
    }
};
```



## 03 majority-element

### python

```python
# two pass + dictionary
def majorityElement1(self, nums):
    dic = {}
    for num in nums:
        dic[num] = dic.get(num, 0) + 1
    for num in nums:
        if dic[num] > len(nums)//2:
            return num
    
# one pass + dictionary
def majorityElement2(self, nums):
    dic = {}
    for num in nums:
        if num not in dic:
            dic[num] = 1
        if dic[num] > len(nums)//2:
            return num
        else:
            dic[num] += 1 

# TLE
def majorityElement3(self, nums):
    for i in xrange(len(nums)):
        count = 0
        for j in xrange(len(nums)):
            if nums[j] == nums[i]:
                count += 1
        if count > len(nums)//2:
            return nums[i]
            
# Sotring            
def majorityElement4(self, nums):
    nums.sort()
    return nums[len(nums)//2]
    
# Bit manipulation    
def majorityElement5(self, nums):
    bit = [0]*32
    for num in nums:
        for j in xrange(32):
            bit[j] += num >> j & 1
    res = 0
    for i, val in enumerate(bit):
        if val > len(nums)//2:
            # if the 31th bit if 1, 
            # it means it's a negative number 
            if i == 31:
                res = -((1<<31)-res)
            else:
                res |= 1 << i
    return res
            
# Divide and Conquer
def majorityElement6(self, nums):
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0]
    a = self.majorityElement(nums[:len(nums)//2])
    b = self.majorityElement(nums[len(nums)//2:])
    if a == b:
        return a
    return [b, a][nums.count(a) > len(nums)//2]
    
# the idea here is if a pair of elements from the
# list is not the same, then delete both, the last 
# remaining element is the majority number
def majorityElement(self, nums):
    count, cand = 0, 0
    for num in nums:
        if num == cand:
            count += 1
        elif count == 0:
            cand, count = num, 1
        else:
            count -= 1
    return cand
```

## 17 letter-combinations-of-a-phone-number

### python

```python
class Solution(object):
    def letterCombinations(self, digits):
        if not digits:
            return []
        mapping = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', 
                   '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        results=['']
        for digit in digits:
            results=[result+d for result in results for d in mapping[digit]]
        return results
```



```python
class Solution(object):
    def letterCombinations(self, digits):
        if not digits:
            return []
        mapping = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', 
                   '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        res=[]
        self.search("",digits,0,res,mapping)
        return res
    def search(self,s,digits,i,res,mapping):
        # recursion terminator 
        if i==len(digits):
            res.append(s)
            return 
        # process logic in current level 
        letter=mapping.get(digits[i])                  
        #show 
        for j in range(len(letter)):
            self.search(s+letter[j],digits,i+1,res,mapping)
        # reverse the current level status if needed
```

### go

```go
var code =[]string{"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"}
func letterCombinations(digits string) []string {
    res:=make([]string,0)
    if len(digits)==0{return res}
    dfs("",digits,&res)
    return res
}
func dfs(cur string,digits string,res *[]string){
    if len(digits)==0{
        *res=append(*res,cur)
        return
    }
    c:=digits[0]-'0'
    for _,v :=range code[c]{
        dfs(cur+string(v),digits[1:],res)
    }
}
```

### C++

```c++
class Solution {
public:
    vector<string> dict = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    vector<string> letterCombinations(string digits) {
        vector<string> res;
        string cur;
        dfs(digits,0,res,cur);
        return res;
    }
    void dfs(string digits,int i,vector<string>& res,string cur){
        if (i ==digits.size()){
            if (!cur.empty()) res.push_back(cur);
        }else{
            for (char c:dict[digits[i]-'0']){
                dfs(digits,i+1,res,cur+c);
            }
        }
    }
};
```



## 51 n-queens

### python

```python
def solveNQueens(self, n):
    def DFS(queens, xy_dif, xy_sum):
        p = len(queens)
        if p==n:
            result.append(queens)
            return None
        for q in range(n):
            if q not in queens and p-q not in xy_dif and p+q not in xy_sum: 
                DFS(queens+[q], xy_dif+[p-q], xy_sum+[p+q])  
    result = []
    DFS([],[],[])
    return [ ["."*i + "Q" + "."*(n-i-1) for i in sol] for sol in result]
```



### go

```go
func solveNQueens(n int) [][]string {
	var res [][]string
	// 生成一个n x n的二维数组，并赋值"."
	board := make([][]rune, n)
	for i := 0; i < n; i++ {
		item := make([]rune, n)
		for j := 0; j < n; j++ {
			item[j] = '.'
		}
		board[i] = item
	}
	backtrack(board, &res, n, 0)
	return res
}

func backtrack(board [][]rune, res *[][]string, n, row int) {
	// 结束条件
	if row == len(board) {
		var arr []string
		for _, v := range board {
			arr = append(arr, string(v))
		}
		// fmt.Println(arr)
		*res = append(*res, arr)
		return
	}
	// 列循环
	for col := 0; col < len(board); col++ {
		// 去除不合法的数据
		if isOk(board, row, col) {
			// 做选择
			board[row][col] = 'Q'
			// 进入下一决策
			backtrack(board, res, n - 1, row + 1)
			// 撤销选择
			board[row][col] = '.'
		}
	}
}

func isOk(board [][]rune, row int, col int) bool {
	// 查询上方数据
	for i := 0; i < row; i++ {
		if board[i][col] == 'Q' {return false}
	}

	// 查询右上角
	for i, j := row-1, col+1; i >= 0 && j < len(board); {
		if board[i][j] == 'Q' {return false}
		i,j=i-1,j+1
	}

	// 查询左上角
	for i, j := row-1, col-1; i >= 0 && j >= 0; {
		if board[i][j] == 'Q' {return false}
		i,j=i-1,j-1
	}
	return true
}
```



### C++

```c++
class Solution
{
public:
    vector<vector<string>> res;
    vector<vector<string>> solveNQueens(int n){
         vector<string> board(n,string(n,'.'));
         backtrack(board, 0);
         return res;
    }

    void backtrack(vector<string>& board,int row){
        if (row==board.size()){
            res.push_back(board);
            return;
        }
        int n=board[row].size();
        for (int col=0;col<n;col++){
            if (!isValid(board, row, col)) continue;
            //做选择
            board[row][col]='Q';
            backtrack(board,row+1);
            board[row][col]='.';
        }
    }

    bool isValid(vector<string>& board,int row,int col){
        int n=board.size();
        for (int i=0;i<n;i++)
            if (board[i][col] =='Q') return false;
        
        for (int i=row-1,j=col+1;i>=0 && j<n;i--,j++)
            if (board[i][j]=='Q') return false;
        for (int i=row-1,j=col-1;i>=0 && j>=0;i--,j--)
            if (board[i][j]=='Q') return false;
        return true;
    }
};
```



```c++
class Solution {
public:
    vector<vector<string> > solveNQueens(int n) {
        vector<vector<string> > res;
        vector<string> nQueens(n, string(n, '.'));
        vector<int> flag_col(n, 1), flag_45(2 * n - 1, 1), flag_135(2 * n - 1, 1);
        solveNQueens(res, nQueens, flag_col, flag_45, flag_135, 0, n);
        return res;
    }
private:
    void solveNQueens(vector<vector<string> > &res, vector<string> &nQueens, vector<int> &flag_col, vector<int> &flag_45, vector<int> &flag_135, int row, int &n) {
        if (row == n) {
            res.push_back(nQueens);
            return;
        }
        for (int col = 0; col != n; ++col)
            if (flag_col[col] && flag_45[row + col] && flag_135[n - 1 + col - row]) {
                flag_col[col] = flag_45[row + col] = flag_135[n - 1 + col - row] = 0;
                nQueens[row][col] = 'Q';
                solveNQueens(res, nQueens, flag_col, flag_45, flag_135, row + 1, n);
                nQueens[row][col] = '.';
                flag_col[col] = flag_45[row + col] = flag_135[n - 1 + col - row] = 1;
            }
    }
};
```



# 深度优先遍历

## 98 validate-binary-search-tree

### python

```python
class Solution(object):
    def isValidBST(self, root):
        res=[]
        self.dfs(root,res)
        for i in range(1,len(res)):
            if res[i-1]>=res[i]:
                return False
        return True
    def dfs(self,root,res):
        if root:
            self.dfs(root.left,res)
            res.append(root.val)
            self.dfs(root.right,res)
```

优化代码

```python
class Solution(object):
    def isValidBST(self, root):
        self.pre=None
        return self.dfs(root)
    def dfs(self,root):
        if not root:
            return True
        if  not self.dfs(root.left):return False
        if self.pre and self.pre.val>=root.val:
            return False
        self.pre=root
        return self.dfs(root.right)
```



```python
class Solution(object):
    def isValidBST(self, root):
        def helper(root,lower=float('-inf'),upper=float('inf')):
            if not root:
                return True
            val=root.val
            if val<=lower or val>=upper:
                return False
            if not helper(root.right,val,upper):
                return False
            if not helper(root.left,lower,val):
                return False
            return True
        return helper(root)
```

优化代码

```python
class Solution(object):
    def isValidBST(self, root):
        def helper(root,lower=float('-inf'),upper=float('inf')):
            if not root: return True
            val=root.val
            if val<=lower or val>=upper: return False
            return helper(root.left,lower,val) and helper(root.right,val,upper)
        return helper(root)
```

### go

```go
func isValidBST(root *TreeNode) bool {
    res:=make([]int,0)
    dfs(root,&res)
    for i:=1;i<len(res);i++{
        if res[i-1]>=res[i]{return false}
    }
    return true
}
func dfs(root *TreeNode,res *[]int){
    if root !=nil{
    dfs(root.Left,res)
    *res=append(*res,root.Val)
    dfs(root.Right,res)
    }
}
```

优化代码

```go
var pre *TreeNode
func isValidBST(root *TreeNode) bool {
	pre = &TreeNode{Val: -1 << 63}
	return dfs(root)
}
func dfs(root *TreeNode) bool {
	if root == nil {
		return true
	}
	if !dfs(root.Left) || pre.Val >=root.Val {
		return false
	}
	pre = root
	return dfs(root.Right)
}
```



```go
func isValidBST(root *TreeNode) bool {
    return dfs(root,math.MinInt64,math.MaxInt64)
}
func dfs(root *TreeNode,min,max int) bool{
    if root ==nil{return true}
    if root.Val<=min || root.Val>=max{return false}
    return dfs(root.Left,min,root.Val) && dfs(root.Right,root.Val,max)
}
```

## 99 recover-binary-search-tree

### python

```python
class Solution(object):
    def recoverTree(self, root):
        self.firstNode = None
        self.secondNode = None
        self.preNode = TreeNode(float("-inf"))
        def dfs(root):
            if not root:return 
            dfs(root.left)
            if self.firstNode==None and self.preNode.val>root.val:
                self.firstNode=self.preNode
            if self.firstNode and self.preNode.val>root.val:
                self.secondNode=root
            self.preNode=root
            dfs(root.right)
        dfs(root)
        self.firstNode.val, self.secondNode.val = self.secondNode.val, self.firstNode.val
```

代码优化

```python
class Solution(object):
    def recoverTree(self, root):
        self.firstNode,self.secondNode,self.preNode = None,None,TreeNode(float("-inf"))
        def dfs(root):
            if root:
                dfs(root.left)
                if self.preNode.val>root.val:
                    self.firstNode=self.firstNode or self.preNode
                    self.secondNode=root
                self.preNode=root
                dfs(root.right)
        dfs(root)
        self.firstNode.val, self.secondNode.val = self.secondNode.val, self.firstNode.val
```

### go

```python
 var pre,first, second *TreeNode
func recoverTree(root *TreeNode)  {
    pre,first,second=nil,nil,nil
    dfs(root)
    first.Val,second.Val=second.Val,first.Val
}
func dfs(root *TreeNode){
    if root ==nil{return}
    dfs(root.Left)
    if pre !=nil && pre.Val>= root.Val{
        if first !=nil{
            second=root
            return
        }
        first,second=pre,root
    }
    pre=root
    dfs(root.Right)
}
```

## 100 same-tree

### python

```python
class Solution(object):
    def isSameTree(self, p, q):
        if not q and not p:return True
        if not q or not p:return False
        if q.val !=p.val:return False
        return self.isSameTree(p.left,q.left) and self.isSameTree(p.right,q.right)
```



```python
class Solution(object):
    def isSameTree(self, p, q):
        queue=deque([(p,q)])
        while queue:
            node1,node2=queue.popleft()
            if not node1 and not node2:continue
            if node1 and node2 and node1.val == node2.val:
                queue.append((node1.left,node2.left))
                queue.append((node1.right,node2.right))
            else:
                return False
        return True
```

```python
class Solution(object):
    def isSameTree(self, p, q):
        queue=deque([(p,q)])
        while queue:
            n1,n2=queue.popleft()
            if not n1 and not n2:continue
            if not n1 or not n2:return n1==n2
            if n1.val !=n2.val:return False
            queue.append((n1.left,n2.left))
            queue.append((n1.right,n2.right))
        return True
```

### go

```go
func isSameTree(p *TreeNode, q *TreeNode) bool {
    if p ==nil && q ==nil{return true}
    if p ==nil && q !=nil {return false}
    if p !=nil && q ==nil{return false}
    if p.Val !=q.Val{return false}
    return isSameTree(p.Left,q.Left) && isSameTree(p.Right,q.Right)
}
```

优化代码

```go
func isSameTree(p *TreeNode, q *TreeNode) bool {
    if p ==nil && q ==nil{return true}
    if p == nil || q == nil || p.Val != q.Val {
        return false
    }
    return isSameTree(p.Left,q.Left) && isSameTree(p.Right,q.Right)
}
```

## 101 symmetric-tree

### python

```python
class Solution(object):
    def isSymmetric(self, root):
        if not root:return True
        def dfs(node1,node2):
            if not node1 and not node2:
                return True
            if node1 and node2:
                if node1.val==node2.val:
                    return dfs(node1.left,node2.right) and dfs(node1.right,node2.left)
            return False
        return dfs(root.left,root.right)
```



```python
class Solution(object):
    def isSymmetric(self, root):
        if not root:return True
        queue=deque([root.left,root.right])
        while queue:
            root_left=queue.pop()
            root_right=queue.pop()
            if not root_left and not root_right:continue
            if  not root_left or not root_right:return False
            if root_left.val !=root_right.val:return False
            queue.extend([root_left.left,root_right.right,root_left.right,root_right.left])
        return True
```



### go

```go
func isSymmetric(root *TreeNode) bool {
    if root ==nil{return true}
    return dfs(root.Left,root.Right)
}
func dfs(root1,root2 *TreeNode) bool{
    if root1 ==nil && root2 ==nil{return true}
    if root1 !=nil && root2 !=nil{
        if root1.Val == root2.Val{
            return dfs(root1.Left,root2.Right) && dfs(root1.Right,root2.Left)
        }
    }
    return false
}
```



```go
func isSymmetric(root *TreeNode) bool {
    if root==nil{return true}
    queue:=[]*TreeNode{root.Left,root.Right}
    for len(queue)>0{
        root_left:=queue[0]
        root_right:=queue[1]
        queue=queue[2:]
        if root_left ==nil && root_right ==nil{
            continue
        }
        if root_left ==nil ||root_right ==nil{
            return false
        }
        if root_left.Val != root_right.Val{
            return false
        }
        queue=append(queue,root_left.Left,root_right.Right)
        queue=append(queue,root_left.Right,root_right.Left)
    }
    return true
}
```



## 105 construct-binary-tree-from-preorder-and-inorder-traversal

```java
public TreeNode buildTree(int[] preorder, int[] inorder) {
    return helper(0, 0, inorder.length - 1, preorder, inorder);
}

public TreeNode helper(int preStart, int inStart, int inEnd, int[] preorder, int[] inorder) {
    if (preStart > preorder.length - 1 || inStart > inEnd) {
        return null;
    }
    TreeNode root = new TreeNode(preorder[preStart]);
    int inIndex = 0; // Index of current root in inorder
    for (int i = inStart; i <= inEnd; i++) {
        if (inorder[i] == root.val) {
            inIndex = i;
        }
    }
    root.left = helper(preStart + 1, inStart, inIndex - 1, preorder, inorder);
    root.right = helper(preStart + inIndex - inStart + 1, inIndex + 1, inEnd, preorder, inorder);
    return root;
}
```



### python

```python
class Solution(object):
    def buildTree(self, preorder, inorder):
        if not preorder:return None
        root=TreeNode(preorder[0])
        i=inorder.index(root.val)
        root.left=self.buildTree(preorder[1:1+i],inorder[:i])
        root.right=self.buildTree(preorder[i+1:],inorder[i+1:])
        return root
```



```python
class Solution(object):
   def buildTree(self, preorder, inorder):
        if inorder:
            ind = inorder.index(preorder.pop(0))
            root = TreeNode(inorder[ind])
            root.left = self.buildTree(preorder, inorder[0:ind])
            root.right = self.buildTree(preorder, inorder[ind+1:])
            return root
```

### go

```go
func buildTree(preorder []int, inorder []int) *TreeNode {
    for k:=range inorder{
        if inorder[k]==preorder[0]{
            return &TreeNode{
                Val:preorder[0],
                Left:buildTree(preorder[1:k+1],inorder[0:k]),
                Right:buildTree(preorder[k+1:],inorder[k+1:]),
            }
        }
    }
    return nil
}
```

```go
func buildTree(preorder []int, inorder []int) *TreeNode {
    if len(inorder)==0{return nil}
    tmp:=-1
    for i,v:=range inorder{
        if v == preorder[0]{
            tmp=i
        }
    }
    root := &TreeNode{Val:preorder[0]}
    root.Left=buildTree(preorder[1:tmp+1],inorder[:tmp])
    root.Right=buildTree(preorder[tmp+1:],inorder[tmp+1:])
    return root
}
```



## 106 construct-binary-tree-from-inorder-and-postorder-traversal

### python

```python
class Solution(object):
    def buildTree(self, inorder, postorder):
        if not inorder:
            return None
        root = TreeNode(postorder[-1])
        i = inorder.index(root.val)
        root.left = self.buildTree(inorder[:i], postorder[:i])
        root.right = self.buildTree(inorder[i+1:], postorder[i:-1])
        return root
```

## 108 convert-sorted-array-to-binary-search-tree

### python

```python
class Solution(object):
    def sortedArrayToBST(self, nums):
        if not nums:return None
        mid=len(nums)>>1
        root=TreeNode(nums[mid])
        left=nums[:mid]
        right=nums[mid+1:]
        root.left=self.sortedArrayToBST(left)
        root.right=self.sortedArrayToBST(right)
        return root
```

### go

```go
func sortedArrayToBST(nums []int) *TreeNode {
    if len(nums) ==0{return nil}
    mid:=len(nums)>>1
    root:=&TreeNode{nums[mid],nil,nil}
    root.Left=sortedArrayToBST(nums[:mid])
    root.Right=sortedArrayToBST(nums[mid+1:])
    return root
}
```

## 109 convert-sorted-list-to-binary-search-tree

### python

```python
class Solution(object):
    def sortedListToBST(self, head):
        nums=[]
        while head:
            nums.append(head.val)
            head=head.next
        def dfs(nums):
            if not nums:return None
            mid=len(nums)>>1
            root=TreeNode(nums[mid])
            root.left=dfs(nums[:mid])
            root.right=dfs(nums[mid+1:])
            return root
        return dfs(nums)
```



```python
class Solution(object):
    def sortedListToBST(self, head):
        return self.dfs(head,None)
    def dfs(self,head,tail):
        if head ==tail:return None
        fast,slow=head,head
        while fast !=tail and fast.next !=tail:
            slow=slow.next
            fast=fast.next.next
        root=TreeNode(slow.val)
        root.left=self.dfs(head,slow)
        root.right=self.dfs(slow.next,tail)
        return root
```

### go

```go
func sortedListToBST(head *ListNode) *TreeNode {
    return dfs(head,nil)
}
func dfs(head,tail ListNode) *TreeNode{
    if head==tail{return nil}
    fast,slow:=head,head
    for fast !=tail && fast.Next !=tail{
        slow=slow.Next
        fast=fast.Next.Next
    }
    root:=&TreeNode{Val:slow.Val}
    root.Left=dfs(head,slow)
    root.Right=dfs(slow.Next,tail)
    return root
}
```

## 110 balanced-binary-tree

### python

```python
class Solution(object):
    def isBalanced(self, root):
        return self.dfs(root) !=-1
    def dfs(self,root):
        if not root:return 0
        left=self.dfs(root.left)
        right=self.dfs(root.right)
        if abs(left-right)>1 or left==-1 or right==-1:
            return -1
        return max(left,right)+1 
```

### go

```go
func isBalanced(root *TreeNode) bool {
    return  dfs(root) !=-1
}
func dfs(root *TreeNode) float64{
    if root ==nil{return 0}
    l:=dfs(root.Left)
    r:=dfs(root.Right)
    if l==-1 || r==-1 || math.Abs(l-r)>1{return -1}
    return math.Max(l,r)+1
}
```

```go
func isBalanced(root *TreeNode) bool {
    return root==nil || isBalanced(root.Left) && 
    math.Abs(height(root.Left)-height(root.Right))<2 && 
    isBalanced(root.Right)
}
func height(root *TreeNode) float64{
    if root ==nil{return 0}
    return 1+math.Max(height(root.Left),height(root.Right))
}
```

## 112 path-sum

### python

```python
class Solution(object):
    def hasPathSum(self, root, sum):
        if not root:
            return False
        sum-=root.val
        if not root.left and  not root.right:
            return sum==0
        return self.hasPathSum(root.left,sum) or self.hasPathSum(root.right,sum)

```

### go

```go
func hasPathSum(root *TreeNode, sum int) bool {
    if root ==nil{return false}
    sum=sum-root.Val
    if root.Left==nil && root.Right==nil{
        return sum==0
    }
    return hasPathSum(root.Left,sum) || hasPathSum(root.Right,sum)
}
```

## 113 path-sum-ii

### python

```python
class Solution(object):
    def pathSum(self, root, sum):
        if not root:
            return []
        res,cur = [],[]
        self.dfs(root, sum, cur, res)
        return res
    
    def dfs(self, root, sum, ls, res):
        if not root:return
        if not root.left and not root.right and sum == root.val:
            ls.append(root.val)
            res.append(ls)
        self.dfs(root.left, sum-root.val, ls+[root.val], res)
        self.dfs(root.right, sum-root.val, ls+[root.val], res)
```

### go

```go
func pathSum(root *TreeNode, sum int) [][]int {
    var res [][]int
    if root == nil {
        return res
    }
    if root.Left == nil && root.Right == nil {
        if sum == root.Val {
            return append(res, []int{ root.Val })
        }
        return res
    }
    
    for _, path := range pathSum(root.Left, sum - root.Val) {
        res = append(res, append([]int{ root.Val}, path... ))
    }
    for _, path := range pathSum(root.Right, sum - root.Val) {
        res = append(res, append([]int{ root.Val}, path... ))
    }
    return res
}
```

```go
func pathSum(root *TreeNode, sum int) [][]int {
	ans := make([][]int, 0)
	path := make([]int, 0)
	helper(root, sum, &ans, path)
	return ans
}

func helper(root *TreeNode, sum int, ans *[][]int, path []int) {
	if root == nil {
		return
	}
	path = append(path, root.Val)
	if root.Left == nil && root.Right == nil && root.Val == sum {
		tmp := make([]int, len(path))
		copy(tmp, path)
		*ans = append(*ans, tmp)
		return
	}
	helper(root.Left, sum-root.Val, ans, path)
	helper(root.Right, sum-root.Val, ans, path)
	path = path[:len(path)-1]  // 回溯，消除上一步造成的影响
}
```

## 114 flatten-binary-tree-to-linked-list

### python

```python
class Solution(object):
    def __init__(self):
        self.pre=None
    def flatten(self, root):
        if not root:return
        self.flatten(root.right)
        self.flatten(root.left)

        root.right=self.pre
        root.left=None
        self.pre=root
```

### go

```go
func flatten(root *TreeNode)  {
    if root ==nil{return}
    flatten(root.Left)
    flatten(root.Right)
    r:=root.Right
    root.Right,root.Left=root.Left,nil
    for root.Right != nil{
        root=root.Right
    }
    root.Right=r
}
```

## 116 populating-next-right-pointers-in-each-node

### python

```python
class Solution(object):
    def connect(self, root):
        if not root:return root
        queue = collections.deque([root])
        while queue:
            size=len(queue)
            for i in range(size):
                node=queue.popleft()
                if i<size-1:
                    node.next=queue[0]
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return root
```

```go
class Solution(object):
    def connect(self, root):
        if not root:
			return root
        pre=root
        while pre.left:
            tmp=pre
            while tmp:
                tmp.left.next=tmp.right
                if tmp.next:
                    tmp.right.next=tmp.next.left
                tmp=tmp.next
            pre=pre.left
        return root
优化代码

```



```python
class Solution(object):
    def connect(self, root):
        if root and root.left and root.right:
            root.left.next = root.right
            if root.next:
                root.right.next = root.next.left
            self.connect(root.left)
            self.connect(root.right)
        return root
```



### go

```go
func connect(root *Node) *Node {
	if root ==nil{return nil}
    pre :=root
    for pre.Left !=nil{
        tmp:=pre
        for tmp!=nil{
            tmp.Left.Next=tmp.Right
            if tmp.Next !=nil{
                tmp.Right.Next=tmp.Next.Left
            }
            tmp=tmp.Next
        }
        pre=pre.Left
    }
    return root
}
```

## 124 binary-tree-maximum-path-sum

### python

```python
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        self.val=float('-inf')
        self.dfs(root)
        return self.val
    def dfs(self,root):
        if not root:return 0
        val=root.val
        left=max(0,self.dfs(root.left))
        right=max(0,self.dfs(root.right))
        self.val=max(self.val,left+right+val)
        return val+max(left,right)
```

### go

```go
func maxPathSum(root *TreeNode) int {
    var res =math.MinInt64
    maxSum(root,&res)
    return res
}
func maxSum(root *TreeNode,maxValue *int)int{
    if root==nil{
        return 0
    }
    val:=root.Val
    left := max(maxSum(root.Left,maxValue),0)
    right :=max( maxSum(root.Right,maxValue),0)
    *maxValue=max(*maxValue,left+right+val)
    return val+max(left,right)
}
func max(x,y int)int{
    if x > y {
        return x
    }
    return y
}
```

## 200 number-of-islands

### python

```python
class Solution(object):
    def numIslands(self, grid):
        if not grid:return 0
        res=0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j]=="1":
                    self.dfs(i,j,grid)
                    res+=1
        return res
    def dfs(self,i,j,grid):
        if i<0 or j<0 or i>=len(grid) or j>=len(grid[0]) or grid[i][j] !='1':
            return 
        grid[i][j]="0"
        self.dfs(i-1,j,grid)
        self.dfs(i+1,j,grid)
        self.dfs(i,j-1,grid)
        self.dfs(i,j+1,grid)
```

### go

```go
func numIslands(grid [][]byte) int {
    if grid ==nil{return 0}
    res:=0
    for i:=0;i<len(grid);i++{
        for j:=0;j<len(grid[0]);j++{
            if grid[i][j] =='1'{
                dfs(i,j,grid)
                res++
            }
        }
    }
    return res
}
func dfs(i,j int,grid [][]byte){
    if i<0 || j<0 || i>=len(grid) ||j>=len(grid[0]) || grid[i][j] !='1'{
        return
    }
    grid[i][j]='0'
    dfs(i-1,j,grid)
    dfs(i+1,j,grid)
    dfs(i,j-1,grid)
    dfs(i,j+1,grid)
}
```



```go
func numIslands(grid [][]byte) int {
    if grid ==nil{return 0}
    res:=0
    for i:=0;i<len(grid);i++{
        for j:=0;j<len(grid[0]);j++{
            if grid[i][j] =='1'{
                dfs(i,j,grid)
                res++
            }
        }
    }
    return res
}

func is_valid(grid [][]byte,r,c int)bool{
    m,n:=len(grid),len(grid[0])
    if r <0 || c<0 ||r>=m || c>=n{
        return false
    }
    return true
}

func dfs(i,j int,grid [][]byte){
   grid[i][j]='0'
   directions:=[][]int{{0,1},{0,-1},{-1,0},{1,0}}
   for _,direction:=range directions{
       x,y:=i+direction[0],j+direction[1]
       if is_valid(grid,x,y) && grid[x][y]=='1'{
           dfs(x,y,grid)
       }
   }
}
```



```go
func numIslands(grid [][]byte) int {
    if grid ==nil{return 0}
    res:=0
    for i:=0;i<len(grid);i++{
        for j:=0;j<len(grid[0]);j++{
            if grid[i][j] =='1'{
                bfs(i,j,grid)
                res++
            }
        }
    }
    return res
}

func is_valid(grid [][]byte,r,c int)bool{
    m,n:=len(grid),len(grid[0])
    if r <0 || c<0 ||r>=m || c>=n{
        return false
    }
    return true
}

func bfs(i,j int,grid [][]byte){
    queue:=[]int{i,j}
   grid[i][j]='0'
   for len(queue)>0{
       i,j:=queue[0],queue[1]
       queue=queue[2:]
        directions:=[][]int{{0,1},{0,-1},{-1,0},{1,0}}
        for _,direction:=range directions{
            x,y:=i+direction[0],j+direction[1]
            if is_valid(grid,x,y) && grid[x][y]=='1'{
                queue=append(queue,x,y)
                grid[x][y]='0'
            }
        }
   }
}
```



# 广度优先遍历

## DFS 代码模板

**递归写法**

```python
visited = set() 

def dfs(node, visited):
    if node in visited: # terminator
    	# already visited 
    	return 

	visited.add(node) 

	# process current node here. 
	...
	for next_node in node.children(): 
		if next_node not in visited: 
			dfs(next_node, visited)
```

**非递归写法**

```python
def DFS(self, tree): 

	if tree.root is None: 
		return [] 

	visited, stack = [], [tree.root]

	while stack: 
		node = stack.pop() 
		visited.add(node)

		process (node) 
		nodes = generate_related_nodes(node) 
		stack.push(nodes) 

	# other processing work 
	...
```

## BFS 代码模板

```python
def BFS(graph, start, end):
    visited = set()
	queue = [] 
	queue.append([start]) 

	while queue: 
		node = queue.pop() 
		visited.add(node)

		process(node)
		nodes = generate_related_nodes(node) 
		queue.push(nodes)
	# other processing work   
```

## 102 binary-tree-level-order-traversal

### python

```python
class Solution(object):
    def levelOrder(self, root):
        if not root: return []
        res,queue=[],collections.deque()
        queue.append(root)
		# visited=set()
        while queue:
            size=len(queue)
            cur_queue=[]
            for _ in range(size):
                node=queue.popleft()
                cur_queue.append(node.val)
                if node.left: queue.append(node.left)
                if node.right:queue.append(node.right)
            res.append(cur_queue)
        return res
```

优化代码

```python
class Solution(object):
    def levelOrder(self, root):
        level,res,queue=0,[],deque([root,])
        if not root:return res
        while queue:
            size=len(queue)
            res.append([])
            for _ in range(size):
                node =queue.popleft()
                res[level].append(node.val)
                if node.left:queue.append(node.left)
                if node.right:queue.append(node.right)
            level+=1
        return res
```





```python
class Solution(object):
    def levelOrder(self, root):
        res=[]
        self.dfs(root,0,res)
        return res
    def dfs(self,root,level,res):
        if not root:return 
        if len(res)==level:res.append([])
        res[level].append(root.val)
        self.dfs(root.left,level+1,res)
        self.dfs(root.right,level+1,res)
```

### go

```go
func levelOrder(root *TreeNode) [][]int {
    var res [][]int
    stack:=[]*TreeNode{}
    if root == nil{return res}
    stack=append(stack,root)
    for len(stack) >0{
        size:=len(stack)
        cur_queueL:=make([]int,0)
        for i:=0;i<size;i++{
            node:=stack[0]
            stack=stack[1:]
            cur_queueL=append(cur_queueL,node.Val)
            if node.Left !=nil{stack=append(stack,node.Left)}
            if node.Right !=nil{stack=append(stack,node.Right)}
        }
        res=append(res,cur_queueL)
    } 
    return res
}
代码优化
func levelOrder(root *TreeNode) [][]int {
    res:=[][]int{}
    if root ==nil{return res}
    var queue []*TreeNode
    queue=append(queue,root)
    level:=0
    for len(queue)>0{
        size:=len(queue)
        res=append(res,[]int{})
        for i:=0;i<size;i++{
            node:=queue[0]
            queue=queue[1:]
            res[level]=append(res[level],node.Val)
            if node.Left !=nil{queue=append(queue,node.Left)}
            if node.Right !=nil{queue=append(queue,node.Right)}
        }
        level++
    }
    return res
}
```

```go
 var res [][]int
func levelOrder(root *TreeNode) [][]int {
    res=make([][]int,0)
    dfs(root,0)
    return res
}
func dfs(root *TreeNode,level int){
    if root ==nil{return}
    if len(res) <level+1{res=append(res,[]int{})}
    res[level]=append(res[level],root.Val)
    dfs(root.Left,level+1)
    dfs(root.Right,level+1)
}
```

### C++

```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        if(root == NULL)
            return {};
        queue<TreeNode*> queue;
        TreeNode* p=root;
        vector<vector<int>> res;
        
        queue.push(p);
        while(!queue.empty()){
            vector<int> temp={};
            // 当前层结点个数
            int length=queue.size();    
            for(int i=0; i<length; i++){
                p=queue.front();
                queue.pop();
                temp.push_back(p->val); 
                // 将当前层结点的子结点入队
                if(p->left) queue.push(p->left);
                if(p->right) queue.push(p->right);
            }
            res.push_back(temp);
        }
        return res;
    }
};
```



```c++
class Solution {
public:
    vector<vector<int>> res;
    vector<vector<int>> levelOrder(TreeNode* root) {
        dfs(root,0);
        return res;
    }
    void dfs(TreeNode* root,int level){
        if (root==NULL) return;
        if (res.size()<level+1) res.push_back(vector<int>());
        res[level].push_back(root->val);
        dfs(root->left,level+1);
        dfs(root->right,level+1);
    }
};
```



## 433 minimum-genetic-mutation

### python

```python
class Solution(object):
    def minMutation(self, start, end, bank):
        queue=collections.deque()
        queue.append((start,0))
        bank=set(bank)
        while queue:
            node,length=queue.popleft()
            if node ==end:
                return length
            for i in range(len(node)):
                for c in "ACGT":
                    new=node[:i]+c+node[i+1:]
                    if new in bank:
                        queue.append((new,length+1))
                        bank.remove(new)
        return -1
```

```python
class Solution:
    def minMutation(self, start, end, bank):
        if end not in bank:
            return -1
        begin = {start}
        end = {end}
        beginStrLen = len(start)
        bank = set(bank)
        length = 0
        while begin and end:
            if len(end) < len(begin):
                end, begin = begin, end
            length += 1
            nextGenSet = set()
            for gen in begin:
                for i in range(beginStrLen):
                    for j in "ACGT":
                        if j == gen[i]:
                            continue
                        nextGen = gen[:i] + j + gen[i + 1:]
                        if nextGen in end:
                            return length
                        if nextGen in bank:
                            nextGenSet.add(nextGen)
                            bank.remove(nextGen)
            begin = nextGenSet
        return -1
```

### C++

双向BFS

```c++
class Solution {
public:
    int minMutation(string start, string end, vector<string>& bank) {
        unordered_set<string> dict(bank.begin(), bank.end());
        if (!dict.count(end)) return -1;
        unordered_set<string> bset, eset, *set1, *set2;
        bset.insert(start), eset.insert(end);
        int step = 0, n = start.size();
        while (!bset.empty() and !eset.empty()) {
            if (bset.size() <= eset.size())
                set1 = &bset, set2 = &eset;
            else set2 = &bset, set1 = &eset;
            unordered_set<string> tmp;
            step ++;
            for (auto itr = set1->begin(); itr != set1->end(); ++itr) {
                for (int i = 0; i < n; ++i) {
                    string dna = *itr;
                    for (auto g : string("ATGC")) {
                        dna[i] = g;
                        if (set2->count(dna)) return step;
                        if (dict.count(dna)) {
                            tmp.insert(dna);
                            dict.erase(dna);
                        }
                    }
                }
            }
            *set1 = tmp;
        }
        return -1;
    }
};
```

0ms

```c++
class Solution {
public:
    int minMutation(string start, string end, vector<string>& bank) {
        queue<string> toVisit;
        unordered_set<string> dict(bank.begin(), bank.end());
        int dist = 0;
        
        if(!dict.count(end)) return -1;
        
        toVisit.push(start);
        dict.insert(start); dict.insert(end);
        while(!toVisit.empty()) {
            int n = toVisit.size();
            for(int i=0; i<n; i++) {
                string str = toVisit.front();
                toVisit.pop();
                if(str==end) return dist;
                addWord(str, dict, toVisit);
            }
            dist++;
        }
        return -1;
        
    }
    
    void addWord(string word, unordered_set<string>& dict, queue<string>& toVisit) {
        dict.erase(word);
        for(int i=0; i<word.size(); i++) {
            char tmp = word[i];
            for(char c:string("ACGT")) {
                word[i] = c;
                if(dict.count(word)) {
                    toVisit.push(word);
                    dict.erase(word);
                }
            }
            word[i] = tmp;
        }
    }
};
```



## 515 find-largest-value-in-each-tree-row

### python

```python
def findValueMostElement(self, root):
    maxes = []
    row = [root]
    while any(row):
        maxes.append(max(node.val for node in row))
        row = [kid for node in row for kid in (node.left, node.right) if kid]
    return maxes
```



```python
def largestValues(self, root):
    if not root:
        return []
    left = self.largestValues(root.left)
    right = self.largestValues(root.right)
    return [root.val] + map(max, left, right)
```

```python
class Solution(object):
    def largestValues(self, root):
        if not root:return []
        res,queue=[],[root]
        while queue:
            res.append(max(node.val for node in queue))
            cur_queue=[]
            for node in queue:
                if node.left:
                    cur_queue.append(node.left)
                if node.right:
                    cur_queue.append(node.right)
            queue=cur_queue
        return res
```

### go

```go
func largestValues(root *TreeNode) []int {
    res:=make([]int,0)
    if root ==nil{return res}
    queue:=make([]*TreeNode,0)
    queue=append(queue,root)
    for len(queue) >0{
        size:=len(queue)
        levelMax := math.MinInt32
        for i:=0;i<size;i++{
            node:=queue[0]
            queue=queue[1:]
            levelMax=int(math.Max(float64(node.Val),float64(levelMax)))
            if node.Left !=nil{queue=append(queue,node.Left)}
            if node.Right !=nil{queue=append(queue,node.Right)}
        }
        res=append(res,levelMax)
    }
    return res
}
```

### C++

```c++
class Solution {
public:
    vector<int> largestValues(TreeNode* root) {
        vector<int> res;
        if(root == NULL) return res;
        queue<TreeNode*> queue;
        queue.push(root);
        while (!queue.empty()){
            int levelMax = INT_MIN;
            int length=queue.size();
            for (int i=0;i<length;i++){
                TreeNode* q=queue.front();
                queue.pop();
                levelMax=max(q->val,levelMax);
                if (q->left) queue.push(q->left);
                if (q->right) queue.push(q->right);
            }
            res.push_back(levelMax);
        }
        return res;
    }
};
```



## 127 word-ladder

### python

单向BFS

```python
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        wordList = set(wordList)
        queue = collections.deque([[beginWord, 1]])
        while queue:
            word, length = queue.popleft()
            if word == endWord:
                return length
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    next_word = word[:i] + c + word[i+1:]
                    if next_word in wordList:
                        wordList.remove(next_word)
                        queue.append([next_word, length + 1])
        return 0
```

双向BFS

```python
import string
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        if endWord not in wordList:return 0
        front={beginWord}
        back={endWord}
        dist=1
        wordList=set(wordList)
        word_len=len(beginWord)
        while front:
            dist+=1
            next_front=set()
            for word in front:
                for i in range(word_len):
                    for c in string.lowercase:
                        new_word=word[:i]+c+word[i+1:]
                        if new_word in back:
                            return dist
                        if new_word in wordList:
                            next_front.add(new_word)
                            wordList.remove(new_word)
            front=next_front
            if len(back)<len(front):
                front,back=back,front
        return 0
```

### go

双向BFS

```go
func ladderLength(beginWord string, endWord string, wordList []string) int {
    dict := make(map[string]bool) // 把word存入字典
    for _, word := range wordList {
        dict[word] = true // 可以利用字典快速添加、删除和查找单词
    }
    if _, ok := dict[endWord]; !ok {
        return 0
    }
    q1 := make(map[string]bool)
    q2 := make(map[string]bool)
    q1[beginWord] = true // 头
    q2[endWord] = true   // 尾

    l := len(beginWord)
    steps := 0

    for len(q1) > 0 && len(q2) > 0 { // 当两个集合都不为空，执行
        steps++
        // Always expend the smaller queue first
        if len(q1) > len(q2) {
            q1, q2 = q2, q1
        }

        q := make(map[string]bool) // 临时set
        for k := range q1 {
            chs := []rune(k)
            for i := 0; i < l; i++ {
                ch := chs[i]
                for c := 'a'; c <= 'z'; c++ { // 对每一位从a-z尝试
                    chs[i] = c // 替换字母组成新的单词
                    t := string(chs)
                    if _, ok := q2[t]; ok { // 看新单词是否在s2集合中
                        return steps + 1
                    }
                    if _, ok := dict[t]; !ok { // 看新单词是否在dict中
                        continue // 不在字典就跳出循环
                    }
                    delete(dict, t) // 若在字典中则删除该新的单词，表示已访问过
                    q[t] = true     // 把该单词加入到临时队列中
                }
                chs[i] = ch // 新单词第i位复位，还原成原单词，继续往下操作
            }
        }
        q1 = q // q1修改为新扩展的q
    }
    return 0
}
```

### C++

```c++
class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> dict(wordList.begin(), wordList.end());
        queue<string> todo;
        todo.push(beginWord);
        int ladder = 1;
        while (!todo.empty()) {
            int n = todo.size();
            for (int i = 0; i < n; i++) {
                string word = todo.front();
                todo.pop();
                if (word == endWord) {
                    return ladder;
                }
                dict.erase(word);
                for (int j = 0; j < word.size(); j++) {
                    char c = word[j];
                    for (int k = 0; k < 26; k++) {
                        word[j] = 'a' + k;
                        if (dict.find(word) != dict.end()) {
                            todo.push(word);
                        }
                     }
                    word[j] = c;
                }
            }
            ladder++;
        }
        return 0;
    }
};
```

```c++
class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> dict(wordList.begin(), wordList.end()), head, tail, *phead, *ptail;
        if (dict.find(endWord) == dict.end()) {
            return 0;
        }
        head.insert(beginWord);
        tail.insert(endWord);
        int ladder = 2;
        while (!head.empty() && !tail.empty()) {
            if (head.size() < tail.size()) {
                phead = &head;
                ptail = &tail;
            } else {
                phead = &tail;
                ptail = &head;
            }
            unordered_set<string> temp;
            for (auto it = phead -> begin(); it != phead -> end(); it++) {    
                string word = *it;
                for (int i = 0; i < word.size(); i++) {
                    char t = word[i];
                    for (int j = 0; j < 26; j++) {
                        word[i] = 'a' + j;
                        if (ptail -> find(word) != ptail -> end()) {
                            return ladder;
                        }
                        if (dict.find(word) != dict.end()) {
                            temp.insert(word);
                            dict.erase(word);
                        }
                    }
                    word[i] = t;
                }
            }
            ladder++;
            phead -> swap(temp);
        }
        return 0;
    }
};
```



## 126 word-ladder-ii

### python

```python
class Solution(object):
    def findLadders(self, beginWord, endWord, wordList):

        wordList = set(wordList)
        res = []
        layer = {}
        layer[beginWord] = [[beginWord]]

        while layer:
            newlayer = collections.defaultdict(list)
            for w in layer:
                if w == endWord: 
                    res.extend(k for k in layer[w])
                else:
                    for i in range(len(w)):
                        for c in 'abcdefghijklmnopqrstuvwxyz':
                            neww = w[:i]+c+w[i+1:]
                            if neww in wordList:
                                newlayer[neww]+=[j+[neww] for j in layer[w]]

            wordList -= set(newlayer.keys())
            layer = newlayer

        return res
```



```python
class Solution(object):
    def findLadders(self,beginWord, endWord, wordList):
        tree, words, n = collections.defaultdict(set), set(wordList), len(beginWord)
        if endWord not in wordList: return []
        found, bq, eq, nq, rev = False, {beginWord}, {endWord}, set(), False
        while bq and not found:
            words -= set(bq)
            for x in bq:
                for y in [x[:i]+c+x[i+1:] for i in range(n) for c in 'qwertyuiopasdfghjklzxcvbnm']:
                    if y in words:
                        if y in eq: 
                            found = True
                        else: 
                            nq.add(y)
                        tree[y].add(x) if rev else tree[x].add(y)
            bq, nq = nq, set()
            if len(bq) > len(eq): 
                bq, eq, rev = eq, bq, not rev
        def bt(x): 
            return [[x]] if x == endWord else [[x] + rest for y in tree[x] for rest in bt(y)]
        return bt(beginWord)
```

# greedy

![1585445453602](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585445453602.png)

![1585445473561](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585445473561.png)

## 01coin-change

### python

```python
class Solution(object):
    def coinChange(self, coins, amount):
        rs = [amount+1] * (amount+1)
        rs[0] = 0
        for i in xrange(1, amount+1):
            for c in coins:
                if i >= c:
                    rs[i] = min(rs[i], rs[i-c] + 1)

        if rs[amount] == amount+1:
            return -1
        return rs[amount]
```

```python
class Solution(object):
    def coinChange(self, coins, amount):
        MAX = float('inf')
        dp = [0] + [MAX] * amount

        for i in xrange(1, amount + 1):
            dp[i] = min([dp[i - c] if i - c >= 0 else MAX for c in coins]) + 1

        return [dp[amount], -1][dp[amount] == MAX]
```



```python
def coinChange(self, coins, amount):
    coins.sort(reverse = True)
    lenc, self.res = len(coins), 2**31-1
    
    def dfs(pt, rem, count):
        if not rem:
            self.res = min(self.res, count)
        for i in range(pt, lenc):
            if coins[i] <= rem < coins[i] * (self.res-count): # if hope still exists
                dfs(i, rem-coins[i], count+1)

    for i in range(lenc):
        dfs(i, amount, 0)
    return self.res if self.res < 2**31-1 else -1
```



### C++

```c++

```



## 02 lemonade-change

### python

```python
class Solution(object):
    def lemonadeChange(self, bills):
        five = ten = 0
        for num in bills:
            if num == 5:five+=1
            elif num ==10:ten,five=ten+1,five-1
            elif ten >0: ten,five=ten-1,five-1
            else: 
                five-=3
            if five <0:return False
        return True
```



```python
class Solution:
    def lemonadeChange(self, bills):
        five = ten = 0
        for num in bills:
            if num == 5:
                five += 1
            elif num == 10 and five:
                ten += 1
                five -= 1
            elif num == 20 and five and ten:
                five -= 1
                ten -= 1
            elif num == 20 and five >= 3:
                five -= 3
            else:
                return False
        return Tru
```



## 03 best-time-to-buy-and-sell-stock-ii

### python

```python
class Solution(object):
    def maxProfit(self, prices):
        return sum(max(prices[i + 1] - prices[i], 0) for i in range(len(prices) - 1))
```



## 04 assign-cookies

### python

```python
class Solution(object):
    def findContentChildren(self, g, s):
        g.sort()
        s.sort()
        
        childi = 0
        cookiei = 0
        
        while cookiei < len(s) and childi < len(g):
            if s[cookiei] >= g[childi]:
                childi += 1
            cookiei += 1
        
        return childi
```



## 05 walking-robot-simulation

### python

```python
class Solution:
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        obst = {tuple(i) for i in obstacles}
        ans = 0
        x, y, dx, dy = 0, 0, 0, 1
        # north: 0, 1
        # south: 0, -1
        # east: 1, 0
        # west: -1, 0
        for i in commands:
            if i == -1:
                dx, dy = dy, -dx
            elif i == -2:
                dx, dy = -dy, dx
            else:
                while i:
                    _x, _y = x + dx, y + dy
                    if (_x, _y) not in obst:
                        x, y, ans = _x, _y, max(ans, _x * _x + _y * _y)
                    i -= 1
        return ans
```



```python
class Solution(object):
    def robotSim(self, commands, obstacles):
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        x = y = di = 0
        obstacleSet = set(map(tuple, obstacles))
        ans = 0

        for cmd in commands:
            if cmd == -2:  #left
                di = (di - 1) % 4
            elif cmd == -1:  #right
                di = (di + 1) % 4
            else:
                for k in xrange(cmd):
                    if (x+dx[di], y+dy[di]) not in obstacleSet:
                        x += dx[di]
                        y += dy[di]
                        ans = max(ans, x*x + y*y)

        return ans
```



## 06problems/jump-game

### python

```python
# DP (like Word Break I) LTE
def canJump1(self, nums):
    dp = [True] * len(nums)
    for i in xrange(1, len(nums)):
        for j in xrange(i):
            dp[i] = dp[j] and nums[j] >= i-j
    return dp[-1]
  
def canJump2(self, nums):
    maxReach = 0
    for i in xrange(len(nums)):
        if i > maxReach:
            return False
        maxReach = max(maxReach, i+nums[i])
    return True
    
def canJump3(self, nums):
    remain = 0
    for i in xrange(len(nums)):
        remain = max(remain-1, nums[i])
        if remain == 0 and i < len(nums)-1:
            return False
    return True
    
def canJump(self, nums):
    maxReach = 0
    i = 0
    while i < len(nums) and i <= maxReach:
        maxReach = max(maxReach, i+nums[i])
        i += 1
    return i == len(nums)
```

## 07jump-game-ii

### python

```python
class Solution:
    def jump(self, nums):
        n, start, end, step = len(nums), 0, 0, 0
        while end < n - 1:
            step += 1
            maxend = end + 1
            for i in range(start, end + 1):
                if i + nums[i] >= n - 1:
                    return step
                maxend = max(maxend, i + nums[i])
            start, end = end + 1, maxend
        return step
```



# 二分查找

## 二分查找代码模板

```python
left, right = 0, len(array) - 1 
while left <= right: 
	  mid = (left + right) / 2 
	  if array[mid] == target: 
		    # find the target!! 
		    break or return result 
	  elif array[mid] < target: 
		    left = mid + 1 
	  else: 
		    right = mid - 1
```

## 64 sqrtx

### python

```python
class Solution(object):
    def mySqrt(self, x):
        r = x
        while r*r > x:
            r = (r + x/r) / 2
        return r
```



```python
# Binary search  
def mySqrt(self, x):
    l, r = 0, x
    while l <= r:
        mid = l + (r-l)//2
        if mid * mid <= x < (mid+1)*(mid+1):
            return mid
        elif x < mid * mid:
            r = mid
        else:
            l = mid + 1
```

## 367 valid-perfect-square

### python

```python
class Solution(object):
    def isPerfectSquare(self, num):
        r=num
        while r*r >num:
            r=(r+num/r)//2
        return r*r==num
```



```python
class Solution(object):
    def isPerfectSquare(self, num):
        if num < 2:
            return True
        left, right = 2, num // 2
        while left <= right:
            x = left + (right - left) // 2
            guess_squared = x * x
            if guess_squared == num:
                return True
            if guess_squared > num:
                right = x - 1
            else:
                left = x + 1
        
        return False
```



## 33 search-in-rotated-sorted-array

### python

```python
class Solution:
    def search(self, nums, target):
        if not nums:
            return -1

        low, high = 0, len(nums) - 1

        while low <= high:
            mid = (low + high) / 2
            if target == nums[mid]:
                return mid

            if nums[low] <= nums[mid]:
                if nums[low] <= target <= nums[mid]:
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                if nums[mid] <= target <= nums[high]:
                    low = mid + 1
                else:
                    high = mid - 1

        return -1
```

# 位运算

![1584583161801](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1584583161801.png)

![1584581051950](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1584581051950.png)

![1584581121258](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1584581121258.png)

![1584581169211](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1584581169211.png)

![1584581240742](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1584581240742.png)



## 191 number-of-1-bits

### python

```python
class Solution(object):
    def hammingWeight(self, n):
        return bin(n).count("1")
```



```python
class Solution(object):
    def hammingWeight(self, n):
        sum=0
        while n !=0:
            sum+=1
            n &=n-1
        return sum
```





## 231 power-of-two

### python

```python
class Solution(object):
    def isPowerOfTwo(self, n):
        return n !=0 and (n & (n-1)) == 0
```



```python
class Solution(object):
    def isPowerOfTwo(self, n):
        return n > 0 and not (n & n-1)
```



```python
class Solution(object):
    def isPowerOfTwo(self, n):
        return n > 0 and bin(n).count('1') == 1
```

## 190 reverse-bits

### python

```python
def reverseBits(self, n):
    ans = 0
    for i in xrange(32):
        ans = (ans << 1) + (n & 1)  #左移一位空出位置 得到的最低位加过来
        n >>= 1 #原数字右移一位去掉已经处理过的最低位
    return ans
```







## 52 N 皇后位运算代码示例

### python

```python
class Solution:
    def totalNQueens(self, n):
        if n < 1: return []
        self.count = 0
        self.DFS(n, 0, 0, 0, 0)
        return self.count

    def DFS(self, n, row, cols, pie, na):
        # recursion terminator
        if row >= n:
            self.count += 1
            return
        bits = (~(cols | pie | na)) & ((1 << n) - 1)  # 得到当前所有的空位

        while bits:
            p = bits & -bits  # 取到最低位的1
            bits = bits & (bits - 1)  # 表示在p位置上放入皇后
            self.DFS(n, row + 1, cols | p, (pie | p) << 1, (na | p) >> 1)
            # 不需要revert  cols, pie, na 的状态
```

# 动态规划

动态规划的要点

![1584928029353](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1584928029353.png)

## 62 unique-paths

### python

```python
class Solution(object):
    def uniquePaths(self, m, n):
        db=[[0]*n for zong in range(m)]
        for i in range(n):db[0][i]=1
        for i in range(m):db[i][0]=1
        for i in range(1,m):
            for j in range(1,n):
                db[i][j]=db[i-1][j]+db[i][j-1]
        return db[m-1][n-1]
```



```python
class Solution:
    def uniquePaths(self, m, n):
        aux = [[1 for x in range(n)] for x in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                aux[i][j] = aux[i][j-1]+aux[i-1][j]
        return aux[-1][-1]
```



```python
# math C(m+n-2,n-1)
def uniquePaths1(self, m, n):
    if not m or not n:
        return 0
    return math.factorial(m+n-2)/(math.factorial(n-1) * math.factorial(m-1))
 
# O(m*n) space   
def uniquePaths2(self, m, n):
    if not m or not n:
        return 0
    dp = [[1 for _ in xrange(n)] for _ in xrange(m)]
    for i in xrange(1, m):
        for j in xrange(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[-1][-1]

# O(n) space 
def uniquePaths(self, m, n):
    if not m or not n:
        return 0
    cur = [1] * n
    for i in xrange(1, m):
        for j in xrange(1, n):
            cur[j] += cur[j-1]
    return cur[-1]
```

## 63 unique-paths-ii

### python

```python
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        if not obstacleGrid:
            return
        width=len(obstacleGrid[0])
        dp=[0]*width
        dp[0]=1
        for i in obstacleGrid:
            for j in range(width):
                if i[j]==1:
                    dp[j]=0
                elif j>0:
                    dp[j]+=dp[j-1]
        return dp[width-1]
```



## 1143 longest-common-subsequence

### python

```python
class Solution(object):
    def longestCommonSubsequence(self, text1, text2):
        if not text1 or not text2:
            return 0
        m=len(text1)
        n=len(text2)
        db=[[0 for _ in range(n+1)] for _ in range(m+1)]
        for i in range(1,m+1):
            for j in range(1,n+1):
                if text1[i-1] == text2[j-1]:
                    db[i][j]=1+db[i-1][j-1]
                else:
                    db[i][j]=max(db[i-1][j],db[i][j-1])
        return db[m][n]
```



```python
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dp = [[0] * (len(text2) + 1) for _ in range(len(text1) + 1)]
        for i, c in enumerate(text1):
            for j, d in enumerate(text2):
                dp[i + 1][j + 1] = 1 + dp[i][j] if c == d else max(dp[i][j + 1], dp[i + 1][j])
        return dp[-1][-1]
```



## 120 triangle

### python

```python
class Solution(object):
    def minimumTotal(self, triangle):
        dp=triangle
        for i in range(len(triangle)-2,-1,-1):
            for j in range(len(triangle[i])):
                dp[i][j]+=min(dp[i+1][j],dp[i+1][j+1])
        return dp[0][0]
```



```python
 # O(n*n/2) space, top-down 
def minimumTotal1(self, triangle):
    if not triangle:
        return 
    res = [[0 for i in xrange(len(row))] for row in triangle]
    res[0][0] = triangle[0][0]
    for i in xrange(1, len(triangle)):
        for j in xrange(len(triangle[i])):
            if j == 0:
                res[i][j] = res[i-1][j] + triangle[i][j]
            elif j == len(triangle[i])-1:
                res[i][j] = res[i-1][j-1] + triangle[i][j]
            else:
                res[i][j] = min(res[i-1][j-1], res[i-1][j]) + triangle[i][j]
    return min(res[-1])
    
# Modify the original triangle, top-down
def minimumTotal2(self, triangle):
    if not triangle:
        return 
    for i in xrange(1, len(triangle)):
        for j in xrange(len(triangle[i])):
            if j == 0:
                triangle[i][j] += triangle[i-1][j]
            elif j == len(triangle[i])-1:
                triangle[i][j] += triangle[i-1][j-1]
            else:
                triangle[i][j] += min(triangle[i-1][j-1], triangle[i-1][j])
    return min(triangle[-1])
    
# Modify the original triangle, bottom-up
def minimumTotal3(self, triangle):
    if not triangle:
        return 
    for i in xrange(len(triangle)-2, -1, -1):
        for j in xrange(len(triangle[i])):
            triangle[i][j] += min(triangle[i+1][j], triangle[i+1][j+1])
    return triangle[0][0]

# bottom-up, O(n) space
def minimumTotal(self, triangle):
    if not triangle:
        return 
    res = triangle[-1]
    for i in xrange(len(triangle)-2, -1, -1):
        for j in xrange(len(triangle[i])):
            res[j] = min(res[j], res[j+1]) + triangle[i][j]
    return res[0] 
```



```python
class Solution(object):
    def minimumTotal(self, triangle):
        if len(triangle)==0:
            return 0
        return self.dfs(0,0,triangle)
    def dfs(self,row,pos,triangle):
            if row+1>=len(triangle):
                return triangle[row][pos]
            left=self.dfs(row+1,pos,triangle)
            right=self.dfs(row+1,pos+1,triangle)
            return triangle[row][pos]+min(left,right)
```

## 53 maximum-subarray

### python

```python
class Solution(object):
    def maxSubArray(self, nums):
        if not nums:
            return 0
        dp=nums
        for i in range(1,len(nums)):
            dp[i]=max(0,dp[i-1])+dp[i]
        return max(dp)
```

## 152 maximum-product-subarray

### python

```python
class Solution(object):
    def maxProduct(self, nums):
        if not nums:
            return 0
        ma=mi=res=nums[0]
        for i in range(1,len(nums)):
            if nums[i] <0:ma,mi=mi,ma
            ma=max(nums[i]*ma,nums[i])
            mi=min(nums[i]*mi,nums[i])
            res=max(ma,res)
        return res
```

## 322 coin-change

### python

#### recursion

```python
class Solution(object):
    def __init__(self):
        self.mem = {0: 0}
    def coinChange(self, coins, amount):
        coins.sort()
        minCoins = self.getMinCoins(coins, amount)
        if minCoins == float('inf'):
            return -1
        return minCoins
    def getMinCoins(self, coins, amount):
        if amount in self.mem:
            return self.mem[amount]
        minCoins = float('inf')
        for c in coins:
            if amount - c < 0:
                break
            numCoins = self.getMinCoins(coins, amount - c) + 1
            minCoins = min(numCoins, minCoins)
        self.mem[amount] = minCoins
        return minCoins
```



#### DFS

```python
def coinChange(self, coins, amount):
    coins.sort(reverse = True)
    lenc, self.res = len(coins), 2**31-1
    def dfs(pt, rem, count):
        if not rem:
            self.res = min(self.res, count)
        for i in range(pt, lenc):
            if coins[i] <= rem < coins[i] * (self.res-count): # if hope still exists
                dfs(i, rem-coins[i], count+1)
    for i in range(lenc):
        dfs(i, amount, 0)
    return self.res if self.res < 2**31-1 else -1
```



#### BFS

```python
class Solution(object):
    def coinChange(self, coins, amount):
        if amount == 0:
            return 0
        value1 = [0]
        value2 = []
        nc =  0
        visited = [False]*(amount+1)
        visited[0] = True
        while value1:
            nc += 1
            for v in value1:
                for coin in coins:
                    newval = v + coin
                    if newval == amount:
                        return nc
                    elif newval > amount:
                        continue
                    elif not visited[newval]:
                        visited[newval] = True
                        value2.append(newval)
            value1, value2 = value2, []
        return -1
```

#### DP

```python
class Solution(object):
    def coinChange(self, coins, amount):
        MAX=float('inf')
        dp=[0]+[MAX]*amount
        for i in range(1,amount+1):
            dp[i]=1+min([dp[i-c] if i-c >=0 else MAX for c in coins])
        return -1 if dp[amount]>amount else dp[amount]
```

## 198 house-robber

### python

```python
class Solution(object):
    def rob(self, nums):
        if not nums:
            return 0
        if len(nums) <= 2:
            return max(nums)
        n=len(nums)
        a=[[0 for x in range(2)] for x in range(n)]
        a[0][0]=0
        a[0][1]=nums[0]
        for i in range(1,n):
            a[i][0]=max(a[i-1][0],a[i-1][1])
            a[i][1]=a[i-1][0]+nums[i]
        return max(a[n-1][0],a[n-1][1])
```





```python
# O(n) space
def rob1(self, nums):
    if not nums:
        return 0
    if len(nums) <= 2:
        return max(nums)
    res = [0] * len(nums)
    res[0], res[1] = nums[0], max(nums[0], nums[1])
    for i in xrange(2, len(nums)):
        res[i] = max(nums[i]+res[i-2], res[i-1])
    return res[-1]

def rob2(self, nums):
    if not nums:
        return 0
    res = [0] * len(nums)
    for i in xrange(len(nums)):
        if i == 0:
            res[0] = nums[0]
        elif i == 1:
            res[1] = max(nums[0], nums[1])
        else:
            res[i] = max(nums[i]+res[i-2], res[i-1])
    return res[-1]
  
# Constant space  
def rob(self, nums):
    if not nums:
        return 0
    if len(nums) <= 2:
        return max(nums)
    a, b = nums[0], max(nums[0], nums[1])
    for i in xrange(2, len(nums)):
        tmp = b
        b = max(nums[i]+a, b)
        a = tmp
    return b
```

```python
class Solution(object):
    def rob(self, nums):
        last, now = 0, 0
        for i in nums: last, now = now, max(last + i, now) 
        return now
```

# Trie

## 基本性质

![1585270023596](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585270023596.png)

## 基本结构

、![1585270132079](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585270132079.png)

![1585270190914](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585270190914.png)

![1585270329848](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585270329848.png)

![1585270396550](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585270396550.png)

## 208 implement-trie-prefix-tree

Trie 树代码模板

```python
class Trie(object):
	def __init__(self): 
		self.root = {} 
		self.end_of_word = "#" 
	def insert(self, word): 
		node = self.root 
		for char in word: 
			node = node.setdefault(char, {}) 
		node[self.end_of_word] = self.end_of_word 
	def search(self, word): 
		node = self.root 
		for char in word: 
			if char not in node: 
				return False 
			node = node[char] 
		return self.end_of_word in node 
	def startsWith(self, prefix): 
		node = self.root 
		for char in prefix: 
			if char not in node: 
				return False 
			node = node[char] 
		return True
```

### go

```go
type Trie struct {
    children [26]*Trie
    isEnd bool
}


/** Initialize your data structure here. */
func Constructor() Trie {
    return Trie{}
}


/** Inserts a word into the trie. */
func (this *Trie) Insert(word string)  {
    curr := this
    for _, ch := range word {
        idx := ch - 'a'
        if curr.children[idx] == nil {
            curr.children[idx] = &Trie{}
        }
        curr = curr.children[idx]
    }
    curr.isEnd = true
}


/** Returns if the word is in the trie. */
func (this *Trie) Search(word string) bool {
    curr := this
    for _, ch := range word {
        idx := ch - 'a'
        if curr.children[idx] == nil {
            return false
        }
        curr = curr.children[idx]
    }
    return curr.isEnd
}


/** Returns if there is any word in the trie that starts with the given prefix. */
func (this *Trie) StartsWith(prefix string) bool {
    curr := this
    for _, ch := range prefix {
        idx := ch - 'a'
        if curr.children[idx] == nil {
            return false
        }
        curr = curr.children[idx]
    }
    return true
}


```



```go
type Trie struct {
	root map[rune]interface{}
}

/** Initialize your data structure here. */
func Constructor() Trie {
	return Trie{root: map[rune]interface{}{}}
}

/** Inserts a word into the trie. */
func (t *Trie) Insert(word string) {
	node := t.root
	for _, c := range word {
		if node[c-'a'] == nil {
			node[c-'a'] = map[rune]interface{}{}
		}
		node = node[c-'a'].(map[rune]interface{})
	}
	// mark End Of Word
	node[26] = struct{}{}
}

/** Returns if the word is in the trie. */
func (t *Trie) Search(word string) bool {
	node := t.root
	for _, c := range word {
		if node[c-'a'] == nil {
			return false
		}
		node = node[c-'a'].(map[rune]interface{})
	}
	// test End Of Word
	return node[26] != nil
}

/** Returns if there is any word in the trie that starts with the given prefix. */
func (t *Trie) StartsWith(prefix string) bool {
	node := t.root
	for _, c := range prefix {
		if node[c-'a'] == nil {
			return false
		}
		node = node[c-'a'].(map[rune]interface{})
	}
	return true
}
```





## 212 word-search-ii

### python

```python
import collections
dx=[-1,1,0,0]
dy=[0,0,-1,1]
END_OF_WORD='#'
class Solution(object):
    def findWords(self, board, words):
        if not board or not board[0]:return []
        if not words:return []
        self.result=set()
        # 构建trie
        root=collections.defaultdict()
        for word in words:
            node=root
            for char in word:
                node=node.setdefault(char,collections.defaultdict())
            node[END_OF_WORD]=END_OF_WORD

        self.m,self.n=len(board),len(board[0])
        for i in range(self.m):
            for j in range(self.n):
                if board[i][j] in root:
                    self._dfs(board,i,j,"",root)
        return list(self.result)
    def _dfs(self,board,i,j,cur_word,cur_dict):
        cur_word+=board[i][j]
        cur_dict=cur_dict[board[i][j]]
        if END_OF_WORD in cur_dict:
            self.result.add(cur_word)
        tmp,board[i][j]=board[i][j],'@'
        for k in range(4):
            x,y=i+dx[k],j+dy[k]
            if 0<=x<self.m and 0<=y<self.n and board[x][y] !='@' and board[x][y] in cur_dict:
                self._dfs(board,x,y,cur_word,cur_dict)
        board[i][j]=tmp
	# 优化代码
     def dfs(self,board,i,j,cur_word,cur_dict):
        cur_word+=board[i][j]
        cur_dict=cur_dict[board[i][j]]
        if END_OF_WORD in cur_dict:
            self.result.add(cur_word)
        tem,board[i][j]= board[i][j],'@'
        directions=[(0,1),(0,-1),(-1,0),(1,0)]
        for direction in directions:
            x,y=i+direction[0],j+direction[1]
            if 0<=x<self.m and 0<=y<self.n and board[x][y] !='@' and board[x][y] in cur_dict:
                self.dfs(board,x,y,cur_word,cur_dict)
        board[i][j]=tem
    

```



```python
class Solution(object):
    def findWords(self, board, words):
        trie={}
        for word in words:
            node=trie
            for char in word:
                node=node.setdefault(char,{})
            node["#"]=True
        
        def dfs(i, j, node, pre, visited):
            if '#' in node:
                res.add(pre)
            for dx,dy in [(0,1),(0,-1),(-1,0),(1,0)]:
                x,y=i+dx,j+dy
                if -1<x<h and -1<y<w and board[x][y] in node and (x,y) not in visited:
                    dfs(x,y,node[board[x][y]],pre+board[x][y],visited | {(x,y)})#并集
       
        res,h,w,=set(),len(board),len(board[0])
        for i in range(h):
            for j in range(w):
                if board[i][j] in trie:
                    dfs(i,j,trie[board[i][j]],board[i][j],{(i,j)})
        return list(res)
```

### go

```go
type TrieNode struct {
	word     string
	children [26]*TrieNode
}

func findWords(board [][]byte, words []string) []string {
	root := &TrieNode{}
	for _, w := range words {
		node := root
		for _, c := range w {
			if node.children[c-'a'] == nil {
				node.children[c-'a'] = &TrieNode{}
			}
			node = node.children[c-'a']
		}
		node.word = w
	}

	result := make([]string, 0)
	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			dfs(i, j, board, root, &result)
		}
	}
	return result
}

func dfs(i, j int, board [][]byte, node *TrieNode, result *[]string) {
	if i < 0 || j < 0 || i == len(board) || j == len(board[0]) {
		return
	}
	c := board[i][j]
	if c == '#' || node.children[c-'a'] == nil {
		return
	}
	node = node.children[c-'a']
	if node.word != "" {
		*result = append(*result, node.word)
		node.word = ""
	}

	board[i][j] = '#'
	dfs(i+1, j, board, node, result)
	dfs(i-1, j, board, node, result)
	dfs(i, j+1, board, node, result)
	dfs(i, j-1, board, node, result)
	board[i][j] = c
}
```



# 并查集

![1585271855678](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585271855678.png)

![1585271928482](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585271928482.png)



![1585272784977](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585272784977.png)

![1585272815965](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585272815965.png)

![1585272892670](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585272892670.png)

![1585273278174](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585273278174.png)

## java 模板

```java
class UnionFind { 
	private int count = 0; 
	private int[] parent; 
	public UnionFind(int n) { 
		count = n; 
		parent = new int[n]; 
		for (int i = 0; i < n; i++) { 
			parent[i] = i;
		}
	} 
	public int find(int p) { 
		while (p != parent[p]) { 
			parent[p] = parent[parent[p]]; 
			p = parent[p]; 
		}
		return p; 
	}
	public void union(int p, int q) { 
		int rootP = find(p); 
		int rootQ = find(q); 
		if (rootP == rootQ) return; 
		parent[rootP] = rootQ; 
		count--;
	}
}
```



## python 模板

```python
class Solution(object):
    def init(self,p):
        p=[i for i in range(n)]
    def union(self,p,i,j):
        p1=self.parent(p,i)
        p2=self.parent(p,j)
        p[p1]=p2
    def parent(self,p,i):
        root=i
        while p[root] !=root:
            root=p[root]
        while p[i] !=i:  #路劲压缩
            x=i;i=p[i];p[x]=root
        return root
```

## 547 friend-circles

### python

```python
class Solution(object):
    def findCircleNum(self, M):
        if not M: return 0
        n=len(M)
        p=[i for i in range(n)]
        for i in range(n):
            for j in range(n):
                if M[i][j] ==1:
                    self.union(p,i,j)
        return len(set([self.parent(p,i) for i in range(n)]))
    def union(self,p,i,j):
        p1=self.parent(p,i)
        p2=self.parent(p,j)
        p[p1]=p2
    def parent(self,p,i):
        root=i
        while p[root] !=root:
            root=p[root]
        while p[i] !=i:  #路劲压缩
            x=i;i=p[i];p[x]=root
        return root
```

```python
class Solution(object):
    def findCircleNum(self, M):
        if not M:
            return 0
        visited = set()
        counter, n = 0, len(M)
        for i in xrange(n):
            if i not in visited:
                self.dfs(M, i, visited)
                counter += 1
        return counter

    def dfs(self, M, i, visited):
        visited.add(i)
        for idx, val in enumerate(M[i]):
            if val == 1 and idx not in visited:
                self.dfs(M, idx, visited)
```

## 200 number-of-islands

### python

```python
def numIslands(self, grid):
    if not grid:
        return 0
        
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == '1':
                self.dfs(grid, i, j)
                count += 1
    return count

def dfs(self, grid, i, j):
    if i<0 or j<0 or i>=len(grid) or j>=len(grid[0]) or grid[i][j] != '1':
        return
    grid[i][j] = '#'
    self.dfs(grid, i+1, j)
    self.dfs(grid, i-1, j)
    self.dfs(grid, i, j+1)
    self.dfs(grid, i, j-1)
```



```python
class Solution(object):
    def is_valid(self, grid, r, c):
        m,n=len(grid),len(grid[0])
        if r<0 or c <0 or r>=m or c>=n:
            return False
        return True
    def numIslands(self, grid):
        if not grid or not grid[0]:return 0
        x,y=len(grid),len(grid[0])
        count=0
        for i in range(x):
            for j in range(y):
                if grid[i][j]=="1":
                    self.dfs(grid,i,j)
                    count+=1
        return count
    def dfs(self,grid,i,j):
        grid[i][j]='0'
        directions=[(0,1),(0,-1),(-1,0),(1,0)]
        for direction in directions:
            x,y=i+direction[0],j+direction[1]
            if self.is_valid(grid,x,y) and grid[x][y]=='1':
                self.dfs(grid,x,y)
```



```
class Solution(object):
    def is_valid(self, grid, r, c):
        m,n=len(grid),len(grid[0])
        if r<0 or c <0 or r>=m or c>=n:
            return False
        return True
    def numIslands(self, grid):
        if not grid or not grid[0]:return 0
        x,y=len(grid),len(grid[0])
        count=0
        for i in range(x):
            for j in range(y):
                if grid[i][j]=="1":
                    self.bfs(grid,i,j)
                    count+=1
        return count
    def bfs(self,grid,i,j):
        queue = collections.deque()
        queue.append((i, j))
        grid[i][j]='0'
        while queue:
            i,j=queue.popleft()
            directions=[(0,1),(0,-1),(-1,0),(1,0)]
            for direction in directions:
                x,y=i+direction[0],j+direction[1]
                if self.is_valid(grid,x,y) and grid[x][y]=='1':
                    queue.append((x,y))
                    grid[x][y]='0'
```





![1586397052343](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1586397052343.png)

## 130 surrounded-regions

### python

```python

```



# 高级搜索

## 剪枝

## 22 generate-parentheses

### python

```phython
class Solution(object):
    def generateParenthesis(self, n):
        res=[]
        self.dfs(0,0,"",n,res)
        return res
    def dfs(self,left,right,path,n,res):
        if left ==n and right==n:
            res.append(path)
            return 
        if left<n:  #剪枝
            self.dfs(left+1,right,path+"(",n,res)
        if left <=n and right < left:  #剪枝
            self.dfs(left,right+1,path+")",n,res)
```

### dp解法

```python
class Solution(object):
    def generateParenthesis(self, n):
        dp=[[] for i in range(n+1)]
        dp[0].append("")
        for i in range(n+1):
            for j in range(i):
                dp[i]+=["("+x+")"+y for x in dp[j] for y in dp[i-j-1]]
        return dp[n]
```

## 36 valid-sudoku

### python

```python
class Solution(object):
    def isValidSudoku(self, board):
        big = set()
        for i in xrange(0,9):
            for j in xrange(0,9):
                if board[i][j]!='.':
                    cur = board[i][j]
                    if (i,cur) in big or (cur,j) in big or (i/3,j/3,cur) in big:
                        return False
                    big.add((i,cur))
                    big.add((cur,j))
                    big.add((i/3,j/3,cur))
        return True
```

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        row = [[x for x in y if x != '.'] for y in board]
        col = [[x for x in y if x != '.'] for y in zip(*board)]
        pal = [[board[i+m][j+n] for m in range(3) for n in range(3) if board[i+m][j+n] != '.'] for i in (0, 3, 6) for j in (0, 3, 6)]
        return all(len(set(x)) == len(x) for x in (*row, *col, *pal))
```

## 37 sudoku-solver

### python

```python
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        row = [set(range(1, 10)) for _ in range(9)]  # 行剩余可用数字
        col = [set(range(1, 10)) for _ in range(9)]  # 列剩余可用数字
        block = [set(range(1, 10)) for _ in range(9)]  # 块剩余可用数字
        
        empty = []  # 收集需填数位置
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':  # 更新可用数字
                    val = int(board[i][j])
                    row[i].remove(val)
                    col[j].remove(val)
                    block[(i // 3)*3 + j // 3].remove(val)
                else:
                    empty.append((i, j))

        def backtrack(iter=0):
            if iter == len(empty):  # 处理完empty代表找到了答案
                return True
            i, j = empty[iter]
            b = (i // 3)*3 + j // 3
            for val in row[i] & col[j] & block[b]:
                row[i].remove(val)
                col[j].remove(val)
                block[b].remove(val)
                board[i][j] = str(val)
                if backtrack(iter+1):
                    return True
                row[i].add(val)  # 回溯
                col[j].add(val)
                block[b].add(val)
            return False
        backtrack()
```



```python
class Solution(object):
    def solveSudoku(self, board):
        def dfs():
            for i, row in enumerate(board):
                for j, char in enumerate(row):
                    if char == '.':
                        for x in s9 - {row[k] for k in r9} - {board[k][j] for k in r9} - \
                        {board[i / 3 * 3 + m][j / 3 * 3 + n] for m in r3 for n in r3}:
                            board[i][j] = x
                            if dfs(): return True
                            board[i][j] = '.'
                        return False
            return True

        r3, r9, s9 = range(3), range(9), {'1', '2', '3', '4', '5', '6', '7', '8', '9'}
        dfs()
```



## 双向 BFS

![1585389287681](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585389287681.png)

![1585389350561](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585389350561.png)

## 127 word-ladder

### python

```python
import string
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        if endWord not in wordList:return 0
        front={beginWord}
        back={endWord}
        dist=1
        wordList=set(wordList)
        word_len=len(beginWord)
        while front:
            dist+=1
            next_front=set()
            for word in front:
                for i in range(word_len):
                    for c in string.lowercase:
                        new_word=word[:i]+c+word[i+1:]
                        if new_word in back:
                            return dist
                        if new_word in wordList:
                            next_front.add(new_word)
                            wordList.remove(new_word)
            front=next_front
            if len(back)<len(front):
                front,back=back,front
        return 0
```

## 启发式搜索

**A*代码模板**

```python
def AstarSearch(graph, start, end):

	pq = collections.priority_queue() # 优先级 —> 估价函数
	pq.append([start]) 
	visited.add(start)

	while pq: 
		node = pq.pop() # can we add more intelligence here ?
		visited.add(node)

		process(node) 
		nodes = generate_related_nodes(node) 
   unvisited = [node for node in nodes if node not in visited]
		pq.push(unvisited)
```

![1585389930923](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585389930923.png)

## 1091 shortest-path-in-binary-matrix

BFS

```python
class Solution(object):
    def shortestPathBinaryMatrix(self, grid):
        q,n=[(0,0,2)],len(grid)  # start end step
        if grid[0][0] or grid[-1][-1]:
            return -1
        if n <=2:return n
        for i,j,d in q:
            for x,y in [(i-1,j-1),(i-1,j),(i-1,j+1),(i,j-1),
                        (i,j+1),(i+1,j-1),(i+1,j),(i+1,j+1)]:
                if 0<=x<n and 0<=y<n and not grid[x][y]:
                    if x ==y==n-1:
                        return d
                    q+=[(x,y,d+1)]
                    grid[x][y]=1
        return -1
```



```python
ss Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if not grid or grid[0][0] == 1 or grid[n-1][n-1] == 1:
            return -1
        elif n <= 2:
            return n
        queue = [(0, 0, 1)]
        grid[0][0] = 1
        while queue:
            i, j, step = queue.pop(0)           
            for dx, dy in [(-1,-1), (1,0), (0,1), (-1,0), (0,-1), (1,1), (1,-1), (-1,1)]:
                if i+dx == n-1 and j+dy == n-1:
                    return step + 1
                if 0 <= i+dx < n and 0 <= j+dy < n and grid[i+dx][j+dy] == 0:
                    queue.append((i+dx, j+dy, step+1))
                    grid[i+dx][j+dy] = 1  # mark as visited                   
        return -1
```



## 773 sliding-puzzle

### python

```python
class Solution(object):
    def slidingPuzzle(self, board):
        board = board[0] + board[1]  # 把board连起来变一维
        moves = [(1, 3), (0, 2, 4), (1, 5), (0, 4), (1, 3, 5), (2, 4)]  # 每个位置的0可以交换的位置
        q, visited = [(tuple(board), board.index(0), 0)], set()  # bfs队列和已访问状态记录
        while q:
            state, now, step = q.pop(0)  # 分别代表当前状态，0的当前位置和当前步数
            if state == (1, 2, 3, 4, 5, 0):  # 找到了
                return step
            for next in moves[now]:  # 遍历所有可交换位置
                _state = list(state)
                _state[next], _state[now] = _state[now], state[next]  # 交换位置
                _state = tuple(_state)
                if _state not in visited:  # 确认未访问
                    q.append((_state, next, step + 1))
            visited.add(state)
        return -1
```



# 红黑树和AVL树

![1585279032337](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585279032337.png)

![1585279089923](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585279089923.png)

![1585279427728](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585279427728.png)

![1585279509167](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585279509167.png)

![1585279563755](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585279563755.png)

![1585279599397](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585279599397.png)

![1585279728267](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585279728267.png)

![1585279976554](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585279976554.png)

![1585280044533](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585280044533.png)

![1585280226203](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1585280226203.png)



# 布隆过滤器的实现及应用

- [布隆过滤器的原理和实现](https://www.cnblogs.com/cpselvis/p/6265825.html)
- [使用布隆过滤器解决缓存击穿、垃圾邮件识别、集合判重](https://blog.csdn.net/tianyaleixiaowu/article/details/74721877)
- [布隆过滤器 Python 代码示例](https://shimo.im/docs/xKwrcwrDxRv3QpKG/)
- [布隆过滤器 Python 实现示例](https://www.geeksforgeeks.org/bloom-filters-introduction-and-python-implementation/)
- [高性能布隆过滤器 Python 实现示例](https://github.com/jhgg/pybloof)
- [布隆过滤器 Java 实现示例      1](https://github.com/lovasoa/bloomfilter/blob/master/src/main/java/BloomFilter.java)
- [布隆过滤器 Java 实现示例      2](https://github.com/Baqend/Orestes-Bloomfilter)

# LRUCache的实现、应用

## 参考链接

- [Understanding the Meltdown exploit](https://www.sqlpassion.at/archive/2018/01/06/understanding-the-meltdown-exploit-in-my-own-simple-words/)
- [替换算法总揽](https://en.wikipedia.org/wiki/Cache_replacement_policies)
- [LRU Cache Python 代码示例](https://shimo.im/docs/tTxRkGwJpXG6WkGY/)

```python
class LRUCache(object): 

	def __init__(self, capacity): 
		self.dic = collections.OrderedDict() 
		self.remain = capacity

	def get(self, key): 
		if key not in self.dic: 
			return -1 
		v = self.dic.pop(key) 
		self.dic[key] = v   # key as the newest one 
		return v 

	def put(self, key, value): 
		if key in self.dic: 
			self.dic.pop(key) 
		else: 
			if self.remain > 0: 
				self.remain -= 1 
			else:   # self.dic is full
				self.dic.popitem(last=False) 
		self.dic[key] = value
```



## 实战题目 / 课后作业

- <https://leetcode-cn.com/problems/lru-cache/#/>

# 排序

## 快排

### python

```python
def quicksort(arr):
    if len(arr)<2:
        return arr
    povit=arr[len(arr)//2]
    left=[x for x in arr if x<povit]
    middle=[x for x in arr if x== povit]
    right=[x for x in arr if x >povit]
    return quicksort(left)+middle+quicksort(right)
print(quicksort([3, 6, 8, 19, 1, 5]))  #[1, 3, 5, 6, 8, 19]
```

## 冒泡排序

### python

```python
def bubble_sort(arr):
    for i in range(len(arr)-1):
        for j in range(len(arr)-i-1):
            if arr[j]>arr[j+1]:
                arr[j],arr[j+1]=arr[j+1],arr[j]
    return arr
print(bubble_sort([3, 6, 8, 19, 1, 5]))  #[1, 3, 5, 6, 8, 19]
```

## 选择排序

### python

```python
def select_sort(arr):
    for i in range(len(arr)-1):
        min_loc=i
        for j in range(i+1,len(arr)):
            if arr[j] <arr[min_loc]:
                min_loc=j
        if min_loc !=i:
            arr[i],arr[min_loc]=arr[min_loc],arr[i]
    return arr
print(select_sort([3, 6, 8, 19, 1, 5]))
```

## 插入排序

### python

````python
def insert_sort(array):
    for i in range(1, len(array)):
        if array[i - 1] > array[i]:
            temp = array[i]     # 当前需要排序的元素
            index = i           # 用来记录排序元素需要插入的位置
            while index > 0 and array[index - 1] > temp:
                array[index] = array[index - 1]     # 把已经排序好的元素后移一位，留下需要插入的位置
                index -= 1
            array[index] = temp # 把需要排序的元素，插入到指定位置
    return array
print(insert_sort([3, 6, 8, 19, 1, 5]))
````

## 归并排序

### python

```python

```

# SQL 语句

```sql
/*
 数据导入：
 Navicat Premium Data Transfer

 Source Server         : localhost
 Source Server Type    : MySQL
 Source Server Version : 50624
 Source Host           : localhost
 Source Database       : sqlexam

 Target Server Type    : MySQL
 Target Server Version : 50624
 File Encoding         : utf-8

 Date: 10/21/2016 06:46:46 AM
*/

SET NAMES utf8;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
--  Table structure for `class`
-- ----------------------------
DROP TABLE IF EXISTS `class`;
CREATE TABLE `class` (
  `cid` int(11) NOT NULL AUTO_INCREMENT,
  `caption` varchar(32) NOT NULL,
  PRIMARY KEY (`cid`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8;

-- ----------------------------
--  Records of `class`
-- ----------------------------
BEGIN;
INSERT INTO `class` VALUES ('1', '三年二班'), ('2', '三年三班'), ('3', '一年二班'), ('4', '二年九班');
COMMIT;

-- ----------------------------
--  Table structure for `course`
-- ----------------------------
DROP TABLE IF EXISTS `course`;
CREATE TABLE `course` (
  `cid` int(11) NOT NULL AUTO_INCREMENT,
  `cname` varchar(32) NOT NULL,
  `teacher_id` int(11) NOT NULL,
  PRIMARY KEY (`cid`),
  KEY `fk_course_teacher` (`teacher_id`),
  CONSTRAINT `fk_course_teacher` FOREIGN KEY (`teacher_id`) REFERENCES `teacher` (`tid`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8;

-- ----------------------------
--  Records of `course`
-- ----------------------------
BEGIN;
INSERT INTO `course` VALUES ('1', '生物', '1'), ('2', '物理', '2'), ('3', '体育', '3'), ('4', '美术', '2');
COMMIT;

-- ----------------------------
--  Table structure for `score`
-- ----------------------------
DROP TABLE IF EXISTS `score`;
CREATE TABLE `score` (
  `sid` int(11) NOT NULL AUTO_INCREMENT,
  `student_id` int(11) NOT NULL,
  `course_id` int(11) NOT NULL,
  `num` int(11) NOT NULL,
  PRIMARY KEY (`sid`),
  KEY `fk_score_student` (`student_id`),
  KEY `fk_score_course` (`course_id`),
  CONSTRAINT `fk_score_course` FOREIGN KEY (`course_id`) REFERENCES `course` (`cid`),
  CONSTRAINT `fk_score_student` FOREIGN KEY (`student_id`) REFERENCES `student` (`sid`)
) ENGINE=InnoDB AUTO_INCREMENT=53 DEFAULT CHARSET=utf8;

-- ----------------------------
--  Records of `score`
-- ----------------------------
BEGIN;
INSERT INTO `score` VALUES ('1', '1', '1', '10'), ('2', '1', '2', '9'), ('5', '1', '4', '66'), ('6', '2', '1', '8'), ('8', '2', '3', '68'), ('9', '2', '4', '99'), ('10', '3', '1', '77'), ('11', '3', '2', '66'), ('12', '3', '3', '87'), ('13', '3', '4', '99'), ('14', '4', '1', '79'), ('15', '4', '2', '11'), ('16', '4', '3', '67'), ('17', '4', '4', '100'), ('18', '5', '1', '79'), ('19', '5', '2', '11'), ('20', '5', '3', '67'), ('21', '5', '4', '100'), ('22', '6', '1', '9'), ('23', '6', '2', '100'), ('24', '6', '3', '67'), ('25', '6', '4', '100'), ('26', '7', '1', '9'), ('27', '7', '2', '100'), ('28', '7', '3', '67'), ('29', '7', '4', '88'), ('30', '8', '1', '9'), ('31', '8', '2', '100'), ('32', '8', '3', '67'), ('33', '8', '4', '88'), ('34', '9', '1', '91'), ('35', '9', '2', '88'), ('36', '9', '3', '67'), ('37', '9', '4', '22'), ('38', '10', '1', '90'), ('39', '10', '2', '77'), ('40', '10', '3', '43'), ('41', '10', '4', '87'), ('42', '11', '1', '90'), ('43', '11', '2', '77'), ('44', '11', '3', '43'), ('45', '11', '4', '87'), ('46', '12', '1', '90'), ('47', '12', '2', '77'), ('48', '12', '3', '43'), ('49', '12', '4', '87'), ('52', '13', '3', '87');
COMMIT;

-- ----------------------------
--  Table structure for `student`
-- ----------------------------
DROP TABLE IF EXISTS `student`;
CREATE TABLE `student` (
  `sid` int(11) NOT NULL AUTO_INCREMENT,
  `gender` char(1) NOT NULL,
  `class_id` int(11) NOT NULL,
  `sname` varchar(32) NOT NULL,
  PRIMARY KEY (`sid`),
  KEY `fk_class` (`class_id`),
  CONSTRAINT `fk_class` FOREIGN KEY (`class_id`) REFERENCES `class` (`cid`)
) ENGINE=InnoDB AUTO_INCREMENT=17 DEFAULT CHARSET=utf8;

-- ----------------------------
--  Records of `student`
-- ----------------------------
BEGIN;
INSERT INTO `student` VALUES ('1', '男', '1', '理解'), ('2', '女', '1', '钢蛋'), ('3', '男', '1', '张三'), ('4', '男', '1', '张一'), ('5', '女', '1', '张二'), ('6', '男', '1', '张四'), ('7', '女', '2', '铁锤'), ('8', '男', '2', '李三'), ('9', '男', '2', '李一'), ('10', '女', '2', '李二'), ('11', '男', '2', '李四'), ('12', '女', '3', '如花'), ('13', '男', '3', '刘三'), ('14', '男', '3', '刘一'), ('15', '女', '3', '刘二'), ('16', '男', '3', '刘四');
COMMIT;

-- ----------------------------
--  Table structure for `teacher`
-- ----------------------------
DROP TABLE IF EXISTS `teacher`;
CREATE TABLE `teacher` (
  `tid` int(11) NOT NULL AUTO_INCREMENT,
  `tname` varchar(32) NOT NULL,
  PRIMARY KEY (`tid`)
) ENGINE=InnoDB AUTO_INCREMENT=6 DEFAULT CHARSET=utf8;

-- ----------------------------
--  Records of `teacher`
-- ----------------------------
BEGIN;
INSERT INTO `teacher` VALUES ('1', '张磊老师'), ('2', '李平老师'), ('3', '刘海燕老师'), ('4', '朱云海老师'), ('5', '李杰老师');
COMMIT;

SET FOREIGN_KEY_CHECKS = 1;
```

```sql
1、查询男生、女生的人数；

2、查询姓“张”的学生名单；

3、课程平均分从高到低显示

4、查询有课程成绩小于60分的同学的学号、姓名；
SELECT sid,sname FROM student RIGHT JOIN (
    SELECT DISTINCT student_id FROM score WHERE num<60) AS t
	ON student.`sid`=t.student_id
SELECT DISTINCT student_id,sname FROM score LEFT JOIN student ON student.`sid`=score.`student_id` WHERE num <60

5、查询至少有一门课与学号为1的同学所学课程相同的同学的学号和姓名；

6、查询出只选修了一门课程的全部学生的学号和姓名；

7、查询各科成绩最高和最低的分：以如下形式显示：课程ID，最高分，最低分；

8、查询课程编号“2”的成绩比课程编号“1”课程低的所有同学的学号、姓名；

9、查询“生物”课程比“物理”课程成绩高的所有学生的学号；

10、查询平均成绩大于60分的同学的学号和平均成绩;

11、查询所有同学的学号、姓名、选课数、总成绩；

12、查询姓“李”的老师的个数；

13、查询没学过“张磊老师”课的同学的学号、姓名；

14、查询学过“1”并且也学过编号“2”课程的同学的学号、姓名；

15、查询学过“李平老师”所教的所有课的同学的学号、姓名；
```



=========================================================================================

# Tree

## python



````python
class TreeNone:
    def __init__(self,val):
        self.val=val
        self.left,self.right=None,None
````



```python
class Tree():
    def __init__(self):#构造出一颗空的二叉树
        self.root = None #root指向第一个节点的地址，如果root指向了None，则意味着该二叉树为空           
    def forward(self,root):
        if root == None:
            return
        print(root.item)
        self.forward(root.left)
        self.forward(root.right)
        
    def middle(self,root):
        if root == None:
            return
        self.middle(root.left)
        print(root.item)
        self.middle(root.right)
    def back(self,root):
        if root == None:
            return
        self.back(root.left)
        self.back(root.right)
        print(root.item)
```



## go



```go
type Tree struct {
	value int
	left *Tree
	right *Tree
} 
```



## java



```java
public class TreeNode{
    public int val;
    public TreeNode left,right;
    public TreeNode(int val){
        this.val=val;
        this.left=null;
        this.right=null;
    }
}
```



## C++



```c++
struct TreeNode{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x):val(x),left(NULL),right(NULL){}
};
```

# 分治代码模板

```python
def divide_conquer(problem, param1, param2, ...): 
  # recursion terminator 
  if problem is None: 
	print_result 
	return 

  # prepare data 
  data = prepare_data(problem) 
  subproblems = split_problem(problem, data) 

  # conquer subproblems 
  subresult1 = self.divide_conquer(subproblems[0], p1, ...) 
  subresult2 = self.divide_conquer(subproblems[1], p1, ...) 
  subresult3 = self.divide_conquer(subproblems[2], p1, ...) 
  …

  # process and generate the final result 
  result = process_result(subresult1, subresult2, subresult3, …)
	
  # revert the current level states
```


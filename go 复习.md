# go 复习

## 退出返回值

Go中main函数不支持任何返回值

可以通过os.Exit()来返回状态

```go
package main//包名

import ( //第三方包
	"fmt"
	"os"
)

func main() {
	fmt.Printf("hello Go")
	os.Exit(-1)
}

```

## 获取命令行参数

与其他主要编程语言的差异

main函数不支持传入参数

在程序中直接通过os.Args获取命令行参数

```go
package main//包名

import ( //第三方包
	"fmt"
	"os"
)

func main() {
	if len(os.Args)>0{
		fmt.Println("hello Go",os.Args[1])
	}
}
```

## 常量和变量

```go
func TestFibList(t *testing.T) {
	// var a int = 1
	// var b int = 1
	// var (
	// 	a int = 1
	// 	b     = 1
	// )
	a := 1
	// a := 1
	b := 1
	t.Log(a)
	for i := 0; i < 5; i++ {
		t.Log(" ", b)
		tmp := a
		a = b
		b = tmp + a
	}

}

func TestExchange(t *testing.T) {
	a := 1
	b := 2
	// tmp := a
	// a = b
	// b = tmp
	a, b = b, a
	t.Log(a, b)
}

```

```go
const (
	Monday = 1 + iota
	Tuesday
	Wednesday
)

const (
	Readable = 1 << iota
	Writable
	Executable
)

func TestConstantTry(t *testing.T) {
	t.Log(Monday, Tuesday)
}

func TestConstantTry1(t *testing.T) {
	a := 1 //0001
	t.Log(a&Readable == Readable, a&Writable == Writable, a&Executable == Executable)
}
```



## 数据类型

bool 

string 

int  int8  int16  int32  int64 uint uint8 uint16 uint32 uint64 uintptr byte // alias for uint8 rune // alias for int32,represents a Unicode code point float32 float64 complex64 complex12

## 类型转化

与其他主要编程语言的差异

1 Go语言不允许隐式类型转化

2 别名和原有类型也不不能进⾏行行隐式类型转换

```go
type MyInt int64

func TestImplicit(t *testing.T) {
	var a int32 = 1
	var b int64
	b = int64(a)
	var c MyInt
	c = MyInt(b)
	t.Log(a, b, c)
}
```

## 类型的预定义值

1. math.MaxInt64 
2. 2. math.MaxFloat64 
   3. 3. math.MaxUint32

## 指针类型

1. 不⽀支持指针运算
2. string 是值类型，其默认的初始化值为空字符串串，⽽而不不是 nil

```go
func TestPoint(t *testing.T) {
	a := 1
	aPtr := &a
	//aPtr = aPtr + 1
	t.Log(a, aPtr)//1 0xc00000a280
	t.Logf("%T %T", a, aPtr)// int *int
}

func TestString(t *testing.T) {
	var s string
	t.Log("*" + s + "*") //初始化零值是“”
	t.Log(len(s))

}
```

## 位运算符

与其他主要编程语⾔言的差异 

&^ 按位置零
1 &^ 0 --  1 

1 &^ 1 --  0 

0 &^ 1 --  0 

0 &^ 0 --  0

```go
const (
	Readable = 1 << iota
	Writable
	Executable
)

func TestBitClear(t *testing.T) {
	a := 7 //0111
	a = a &^ Readable
	a = a &^ Executable
	t.Log(a&Readable == Readable, a&Writable == Writable, a&Executable == Executable)
}
```

## switch条件

1. 条件表达式不不限制为常量量或者整数； 
2. 单个 case 中，可以出现多个结果选项, 使⽤用逗号分隔； 
3. 与 C 语⾔言等规则相反，Go 语⾔言不不需要⽤用break来明确退出⼀一个 case； 
4. 可以不不设定 switch 之后的条件表达式，在此种情况下，整个 switch 结 构与多个 if…else… 的逻辑作⽤用等同

```go
func TestSwitchMultiCase(t *testing.T) {
	for i := 0; i < 5; i++ {
		switch i {
		case 0, 2:
			t.Log("Even")
		case 1, 3:
			t.Log("Odd")
		default:
			t.Log("it is not 0-3")
		}
	}
}

func TestSwitchCaseCondition(t *testing.T) {
	for i := 0; i < 5; i++ {
		switch {
		case i%2 == 0:
			t.Log("Even")
		case i%2 == 1:
			t.Log("Odd")
		default:
			t.Log("unknow")
		}
	}
}
```

## 切⽚片声明

var s0 []int 

s0 = append(s0, 1) 

s := []int{} 

s1 := []int{1, 2, 3} 

s2 := make([]int, 2, 4)   /*[]type, len, cap  其中len个元素会被初始化为默认零值，未初始化元素不不可以访问  */	

## Map 声明

m := map[string]int{"one": 1, "two": 2, "three": 3} 

m1 := map[string]int{} 

m1["one"] = 1 

m2 := make(map[string]int, 10 /*Initial Capacity*/) //为什什么不不初始化len？

在访问的 Key 不不存在时，仍会返回零值，不不能通过返回 nil 来判断元素是否存在

```go
if v, ok := m["four"]; ok { 
    t.Log("four", v) 
} else {  
    t.Log("Not existing") 
}

```

## 常⽤用字符串串函数

```go
func TestStringFn(t *testing.T) {
	s := "A,B,C"
	parts := strings.Split(s, ",")
	for _, part := range parts {
		t.Log(part)
	}
	t.Log(strings.Join(parts, "-"))
}

func TestConv(t *testing.T) {
	s := strconv.Itoa(10)//转化为字符
	t.Log("str" + s)
	if i, err := strconv.Atoi("10"); err == nil {//转为数字
		t.Log(10 + i)
	}
}

```



​	

## Map 与⼯工⼚厂模式

```go
func TestMapWithFunValue(t *testing.T)  {
	m:=map[int]func(op int)int{}
	m[1]= func(op int) int {return op}
	m[2]= func(op int) int {return op*op}
	m[3]= func(op int) int {return op*op*op}
	t.Log(m[1](2),m[2](2),m[3](2))
}
```

## 字符串常用函数

```go
func TestStringFn(t *testing.T)  {
	s:="A,B,C"
	part:=strings.Split(s,",")
	for _,v:=range part{
		t.Log(v)
	}
	t.Log(strings.Join(part,"_"))
}
func TestConv(t *testing.T)  {
	s:=strconv.Itoa(10)//转换为字符
	t.Log("str "+s)
	if i,err:=strconv.Atoi("10");err==nil{//转化为数字
		t.Log(10+i)
	}
}

```

## 函数

1. 可以有多个返回值
2. 所有参数都是值传递：slice，map，channel 会有传引⽤用的错觉
3. 函数可以作为变量量的值
4. 函数可以作为参数和返回值

```go
func Clear() {
	fmt.Println("Clear resources.")
}

func TestDefer(t *testing.T) {
	defer Clear()
	fmt.Println("Start")
	panic("err")  //defer仍会执⾏行行
}
```

```go
type IntConv func(op int) int   //别名

func timeSpent(inner IntConv) IntConv {
	return func(n int) int {
		start := time.Now()
		ret := inner(n)
		fmt.Println("time spent:", time.Since(start).Seconds())
		return ret
	}
}
```

## 面向对象

```go
type Programmer interface {
	WriteHelloWorld() string
}

type GoProgrammer struct {
}

func (g *GoProgrammer) WriteHelloWorld() string {
	return "fmt.Println(\"Hello World\")"
}

func TestClient(t *testing.T) {
	var p Programmer
	p = new(GoProgrammer)
	t.Log(p.WriteHelloWorld())
}

```

## 错误处理

```go
var LessThanTwoError = errors.New("n should be not less than 2")
var LargerThenHundredError = errors.New("n should be not larger than 100")

func GetFibonacci(n int) ([]int, error) {
	if n < 2 {
		return nil, LessThanTwoError
	}
	if n > 100 {
		return nil, LargerThenHundredError
	}
	fibList := []int{1, 1}

	for i := 2; /*短变量声明 := */ i < n; i++ {
		fibList = append(fibList, fibList[i-2]+fibList[i-1])
	}
	return fibList, nil
}
```

## 协程

```go
func TestCounter(t *testing.T) {

	counter := 0
	for i := 0; i < 5000; i++ {
		go func() {
			counter++
		}()
	}
	time.Sleep(1 * time.Second)
	t.Logf("counter = %d", counter)

}

func TestCounterThreadSafe(t *testing.T) {
	var mut sync.Mutex
	counter := 0
	for i := 0; i < 5000; i++ {
		go func() {
			defer func() {
				mut.Unlock()
			}()
			mut.Lock()
			counter++
		}()
	}
	time.Sleep(1 * time.Second)
	t.Logf("counter = %d", counter)

}

func TestCounterWaitGroup(t *testing.T) {
	var mut sync.Mutex
	var wg sync.WaitGroup
	counter := 0
	for i := 0; i < 5000; i++ {
		wg.Add(1)
		go func() {
			defer func() {
				mut.Unlock()
			}()
			mut.Lock()
			counter++
			wg.Done()
		}()
	}
	wg.Wait()
	t.Logf("counter = %d", counter)

}
```

## Channel

```go
func service() string {
	time.Sleep(time.Millisecond * 50)
	return "Done"
}

func otherTask() {
	fmt.Println("working on something else")
	time.Sleep(time.Millisecond * 100)
	fmt.Println("Task is done.")
}

func TestService(t *testing.T) {
	fmt.Println(service())
	otherTask()
}

func AsyncService() chan string {
	retCh := make(chan string, 1)
	//retCh := make(chan string, 1)
	go func() {
		ret := service()
		fmt.Println("returned result.")
		retCh <- ret
		fmt.Println("service exited.")
	}()
	return retCh
}

//
func TestAsynService(t *testing.T) {
	retCh := AsyncService()
	otherTask()
	fmt.Println(<-retCh)
	time.Sleep(time.Second * 1)
}

```

## select

```go
func service() string {
	time.Sleep(time.Millisecond * 500)
	return "Done"
}

func AsyncService() chan string {
	retCh := make(chan string, 1)
	//retCh := make(chan string, 1)
	go func() {
		ret := service()
		fmt.Println("returned result.")
		retCh <- ret
		fmt.Println("service exited.")
	}()
	return retCh
}

func TestSelect(t *testing.T) {
	select {
	case ret := <-AsyncService():
		t.Log(ret)
	case <-time.After(time.Millisecond * 100)://超时控制
		t.Error("time out")
	}
}

```

## channel 的关闭和⼴广播

• 向关闭的 channel 发送数据，会导致 panic
• v, ok <-ch; ok 为 bool 值，true 表示正常接受，false 表示通道关闭
• 所有的 channel 接收者都会在 channel 关闭时，⽴立刻从阻塞等待中返回且上 述 ok 值为 false。这个⼴广播机制常被利利⽤用，进⾏行行向多个订阅者同时发送信号。 如：退出信号。

```go
func dataProducer(ch chan int, wg *sync.WaitGroup) {
	go func() {
		for i := 0; i < 10; i++ {
			ch <- i
		}
		close(ch)

		wg.Done()
	}()

}

func dataReceiver(ch chan int, wg *sync.WaitGroup) {
	go func() {
		for {
			if data, ok := <-ch; ok {
				fmt.Println(data)
			} else {
				break
			}
		}
		wg.Done()
	}()

}

func TestCloseChannel(t *testing.T) {
	var wg sync.WaitGroup
	ch := make(chan int)
	wg.Add(1)
	dataProducer(ch, &wg)
	wg.Add(1)
	dataReceiver(ch, &wg)
	// wg.Add(1)
	// dataReceiver(ch, &wg)
	wg.Wait()

}
```

## 任务的取消

```go
func isCancelled(cancelChan chan struct{}) bool {  //获取取消通知
	select {
	case <-cancelChan:
		return true
	default:
		return false
	}
}

func cancel_1(cancelChan chan struct{}) {
	cancelChan <- struct{}{}//发送取消消息
}

func cancel_2(cancelChan chan struct{}) {
	close(cancelChan)//通过关闭 Channel 取消
}

func TestCancel(t *testing.T) {
	cancelChan := make(chan struct{}, 0)
	for i := 0; i < 5; i++ {
		go func(i int, cancelCh chan struct{}) {
			for {
				if isCancelled(cancelCh) {
					break
				}
				time.Sleep(time.Millisecond * 5)
			}
			fmt.Println(i, "Cancelled")
		}(i, cancelChan)
	}
	cancel_2(cancelChan)
	time.Sleep(time.Second * 1)
}

```

## Context 与任务取消

Context 

• 根 Context：通过 context.Background () 创建
• ⼦子 Context：context.WithCancel(parentContext) 创建
• ctx, cancel := context.WithCancel(context.Background())
• 当前 Context 被取消时，基于他的⼦子 context 都会被取消
• 接收取消通知 <-ctx.Done()

```go
func isCancelled(ctx context.Context) bool {
	select {
	case <-ctx.Done():
		return true
	default:
		return false
	}
}

func TestCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	for i := 0; i < 5; i++ {
		go func(i int, ctx context.Context) {
			for {
				if isCancelled(ctx) {
					break
				}
				time.Sleep(time.Millisecond * 5)
			}
			fmt.Println(i, "Cancelled")
		}(i, ctx)
	}
	cancel()
	time.Sleep(time.Second * 1)
}
```

## 常见并发任务

```go
type Singleton struct {
	data string
}

var singleInstance *Singleton
var once sync.Once

func GetSingletonObj() *Singleton {
	once.Do(func() {
		fmt.Println("Create Obj")
		singleInstance = new(Singleton)
	})
	return singleInstance
}

func TestGetSingletonObj(t *testing.T) {
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			obj := GetSingletonObj()
			fmt.Printf("%X\n", unsafe.Pointer(obj))
			wg.Done()
		}()
	}
	wg.Wait()
}
```


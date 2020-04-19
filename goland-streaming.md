# Go语言实战流媒体视频网站

![1587171159979](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1587171159979.png)

## 什么是前后端解耦

- 前后端解耦是时下流行的 web 网站架构
- 前端页面和服务通过普通的 web 引擎渲染
- 后端数据通过渲染后的页面脚本调用后处理和呈现

## 前后端解耦的优势

- 解放生产力，提高合作效率
- 松耦合的架构更灵活，部署更方便，更符合微服务的设计特征
- 性能的提升，可靠性的提升

## 前后端解耦的缺点

- 工作量大
- 前后端分享带来的团队成本以及学习成本
- 系统复杂度加大

## API

- REST(Representational State Transfer) API
- REST是一种设计风格，不是任何架构标准
- 当今 RESTful API通常使用 HTTP 作为通信协议，JSON 作为数据格式
- 特点
  - 统一接口（Uniform Interface）
  - 无状态（Stateless）
  - 可缓存（Cachable）
  - 分层（Layered System）
  - CS 模式（Client-server Architecture）
- 设计原则创建（注册）用户：URL: /user Method: POST, SC: 201, 400, 500
- 用户登录：URL: /user/:username Method: POST, SC: 200, 400, 500
- 获取用户基本信息：URL: /user/:username Method: GET, SC: 200, 400, 401, 403, 500
- 用户注销：URL: /user/:username Method: DELETE, SC: 204, 400, 401, 403, 500

goroutine 是非常轻量级的协程，每个协程仅占用4K内存

listen -> RegisterHandlers -> handlers

- 
  - 以 URL（统一资源定位符）风格设计 API
  - 通过不同的 METHOD（GET, POST, PUT, DELETE）来区分对资源的 CRUD
  - 返回码（Status Code）符合 HTTP 资源描述的规定

[![Go语言实战流媒体视频网站](https://alanhou.org/homepage/wp-content/uploads/2019/03/2019040215022895.jpg)](https://alanhou.org/homepage/wp-content/uploads/2019/03/2019040215022895.jpg)

### API设计：用户

- 创建（注册）用户：URL: /user Method: POST, SC: 201, 400, 500
- 用户登录：URL: /user/:username Method: POST, SC: 200, 400, 500
- 获取用户基本信息：URL: /user/:username Method: GET, SC: 200, 400, 401, 403, 500
- 用户注销：URL: /user/:username Method: DELETE, SC: 204, 400, 401, 403, 500

goroutine 是非常轻量级的协程，每个协程仅占用4K内存

listen -> RegisterHandlers -> handlers

### API设计：用户资源

- List all videos：URL:/user/:username/videos Method: GET, SC: 200, 400, 500
- Get one video：URL:/user/:/username/videos/:vid-id Method: GET, SC: 200, 400, 500
- Delete on video： URL:/user/:username/videos/:vid-id Method: GET, SC: 204, 400, 401, 403, 500

### API设计：评论

- Show comments：URL:/videos/:vid-id/comments Method: GET, SC: 200, 400, 500
- Post a comment：URL:/videos/:vid-id/comments Method: POST, SC: 201, 400, 500
- Delete a comment：URL:/videos/:vid-id/comment/:comment-id Method: DELETE, SC: 204, 400, 401, 403, 500

![1587174009557](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1587174009557.png)

```
//handler->validation{1.request,2.user}->business logic->response
//1 data model
//2 error handling
```

![1587174807772](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1587174807772.png)

## 数据库设计

### 数据库设计：用户

```sql
TABLE: users
id  INT UNSIGNED PRIMARY KEY AUTO_INCREMENT
login_name VARCHAR(64) UNIQUE KEY
pwd TEXT
```

### 数据库设计：视频资源

```sql
TABLE: video_info
id VARCHAR(64) PRIMARY KEY NOT NULL
author_id  INT UNSIGNED
name TEXT
display_ctime TEXT
create_time DATETIME
```

### 数据库设计：评论

```sql
TABLE: comments
id VARCHAR(64) PRIMARY NOT NULL
video_id VARCHAR(64)
author_id INT UNSIGNED
content TEXT
time DATETIME
```

### 数据库设计：sessions

```sql
TABLE: sessions
session_id VARCHAR(255) PRIMARY KEY NOT NULL
TTL TINYTEXT
login_name VARCHAR(64)
```

### Session

- 什么是 session
- 为什么要用 session
- session 和 cookie 的区别：服务端&客户端

## 数据库功能

初始化数据conn.go

```go
package dbops

import (
	"database/sql"
	_ "github.com/go-sql-driver/mysql"
)
var (
	dbConn *sql.DB
	err error
)

func init()  {
	dbConn,err=sql.Open("mysql","root:123456@tcp(127.0.0.1:3306)/video_server?charset=utf8")
	if err !=nil{
		panic(err.Error())
	}
}
```

### 用户数据库操作

```go
//增加用户
func AddUserCredential(loginName string,pwd string)error  {
	stmtIns,err:=dbConn.Prepare("insert into users (login_name,pwd) values (?,?)")//Prepare 预编译
	if err !=nil{
		return err
	}

	stmtIns.Exec(loginName,pwd)
	stmtIns.Close()
	return nil
}
//查询用户
func GetUserCredential(loginName string)(string,error)  {
	stmtOut,err:=dbConn.Prepare("select pwd from users where login_name=?")
	if err !=nil{
		log.Printf("%s",err)
		return "",err
	}

	var pwd string
	stmtOut.QueryRow(loginName).Scan(&pwd)
	stmtOut.Close()

	return pwd,nil
}
//删除用户
func DeleteUser(loginName,pwd string)error  {
	stmtDel,err:=dbConn.Prepare("delete from users where login_name=? and pwd=?")
	if err !=nil{
		log.Printf("%s",err)
		return err
	}

	stmtDel.Exec(loginName,pwd)
	stmtDel.Close()
	return nil
}
```

### 测试程序

```go
package dbops

import "testing"

//init(dblogin, truncate tables) -> run tests -> clear data(truncate tables)
func clearTables()  {
	dbConn.Exec("truncate users")
	dbConn.Exec("truncate video_info")
	dbConn.Exec("truncate comments")
	dbConn.Exec("truncate sessions")
}
func TestMain(m *testing.M)  {
	clearTables()
	m.Run()
	clearTables()
}
func TestUserWorkFlow(t *testing.T)  {
	t.Run("Add",testAddUser)
	t.Run("Get",testGetUser)
	t.Run("Del",testDeleteUser)
	t.Run("Reget",testRegetUser)
}
func testAddUser(t *testing.T)  {
	err:=AddUserCredential("avenssi","123")
	if err !=nil{
		t.Errorf("Error of AddUser: %v",err)
	}
}
func testGetUser(t *testing.T)  {
	pwd,err:=GetUserCredential("avenssi")
	if pwd !="123"||err !=nil{
		t.Errorf("Error of GetUser")
	}
}
func testDeleteUser(t *testing.T)  {
	err:=DeleteUser("avenssi","123")
	if err !=nil{
		t.Errorf("Error of DeleteUser: %v", err)
	}
}
func testRegetUser(t *testing.T)  {
	pwd,err:=GetUserCredential("avenssi")
	if err != nil {
		t.Errorf("Error of RegetUser: %v", err)
	}

	if pwd != "" {
		t.Errorf("Deleting user test failed")
	}
}
```

代码优化

```go
func AddUserCredential(loginName string,pwd string)error  {
	stmtIns,err:=dbConn.Prepare("insert into users (login_name,pwd) values (?,?)")//Prepare 预编译
	if err !=nil{
		return err
	}

	_,err=stmtIns.Exec(loginName,pwd)
	if err!=nil{
		return err
	}
	defer stmtIns.Close()
	return nil
}
//查询用户
func GetUserCredential(loginName string)(string,error)  {
	stmtOut,err:=dbConn.Prepare("select pwd from users where login_name=?")
	if err !=nil{
		log.Printf("%s",err)
		return "",err
	}

	var pwd string
	err=stmtOut.QueryRow(loginName).Scan(&pwd)
	if err!=nil && err !=sql.ErrNoRows{
		return "",err
	}
	defer stmtOut.Close()

	return pwd,nil
}
//删除用户
func DeleteUser(loginName,pwd string)error  {
	stmtDel,err:=dbConn.Prepare("delete from users where login_name=? and pwd=?")
	if err !=nil{
		log.Printf("%s",err)
		return err
	}

	_,err=stmtDel.Exec(loginName,pwd)
	if err !=nil{
		return err
	}
	defer stmtDel.Close()
	return nil
}
```

# 一个基于 Go 语言实现的分布式云存储服务，慕课网实战仿百度网盘项目。

![1587287050194](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1587287050194.png)



![1587287190691](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1587287190691.png)

1



1

![1587286956684](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1587286956684.png)

![1587287428893](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1587287428893.png)

![1587287652080](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1587287652080.png)

![1587287708297](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1587287708297.png)

![1587287987525](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1587287987525.png)

![1587289714086](C:\Users\建祥\AppData\Roaming\Typora\typora-user-images\1587289714086.png)


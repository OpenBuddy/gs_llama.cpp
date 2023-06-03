package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"runtime"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/gorilla/websocket"
)

// #cgo CXXFLAGS: -I.
// #cgo LDFLAGS: -L../../build -L../../build/examples/gsnode -lstdc++ -lm -lgs -lllama
// #include "gs.h"
import "C"

var flagModelPath = flag.String("model", "/mnt/c/AITemp/7b-v1.3-q5_1.bin", "Path to the model file")
var flagServer = flag.String("server", "ws://127.0.0.1:8120", "server url")
var flagMaxConcurrency = flag.Int("maxc", 1, "max concurrency")
var flagNodeName = flag.String("name", "beagle", "node name")
var flagToken = flag.String("token", "unsafe-default-token", "token")
var modelName = ""

type ChatMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type Task struct {
	ID           uint32    `json:"id"`
	MaxNewTokens int       `json:"max_new_tokens"`
	System       string    `json:"system"`
	Messages     []ChatMsg `json:"messages"`
	Temperature  float64   `json:"temperature"`
	Stop         bool      `json:"stop"`

	// For internal use
	tmpBuf []byte `json:"-"`
}

var reqChan = make(chan *Task, 1000)
var respChan = make(chan []byte, 1000)
var currentTasks = make(map[uint32]*Task)

func wsConnectAndHandle(url string) {
	// Connect to ws url
	c, _, err := websocket.DefaultDialer.Dial(url, nil)
	if err != nil {
		log.Println("dial:", err)
		return
	}
	defer c.Close()
	go func() {
		// Read from ws
		for {
			t, m, err := c.ReadMessage()
			if err != nil {
				log.Println("[ws] read:", err)
				c.Close()
				return
			}
			log.Println("[ws] recv: ", t, string(m))
			// If is string message, handle it
			if t == websocket.TextMessage {
				reqObj := Task{}
				err := json.Unmarshal(m, &reqObj)
				if err != nil {
					log.Println("[ws] unmarshal:", err)
					continue
				}
				reqChan <- &reqObj
			}
		}
	}()
	for {
		select {
		case resp := <-respChan:
			// Send to ws
			err = c.WriteMessage(websocket.BinaryMessage, resp)
			if err != nil {
				log.Println("[ws] write:", err)
				return
			}
		case <-time.After(30 * time.Second):
			// Send ping
			err = c.WriteMessage(websocket.TextMessage, []byte(""))
			if err != nil {
				log.Println("[ws] write ping:", err)
				return
			}
		}
	}

}

func wsLoop() {

	url := fmt.Sprintf("%s/ws?name=%s&model=%s&token=%s&max_concurrency=%d", *flagServer, *flagNodeName, modelName, *flagToken, *flagMaxConcurrency)
	log.Println("URL: ", url)
	for {
		wsConnectAndHandle(url)
		log.Println("Waiting 30 seconds to reconnect...")
		time.Sleep(30 * time.Second)
	}
}

func main() {
	modelName = *flagModelPath
	// Remove ".bin" suffix
	modelName = strings.TrimSuffix(modelName, ".bin")
	// Remove before "/" or "\\"
	if strings.Contains(modelName, "/") {
		modelName = modelName[strings.LastIndex(modelName, "/")+1:]
	}
	if strings.Contains(modelName, "\\") {
		modelName = modelName[strings.LastIndex(modelName, "\\")+1:]
	}

	runtime.LockOSThread()
	flag.Parse()
	C.gsInit(C.CString(*flagModelPath))
	go wsLoop()

	for {
		select {
		case req := <-reqChan:
			if req.Stop {
				FinishTask(req.ID)
			} else {
				handleNewTask(req)
			}
		case <-time.After(5 * time.Millisecond):
			C.gsDoOnce()
		}

	}

}

func handleNewTask(req *Task) {

	sb := strings.Builder{}
	if req.Messages == nil {
		return
	}
	if len(req.Messages) <= 0 {
		return
	}
	for _, msg := range req.Messages {
		isAssistant := false
		r := "User: "
		if strings.ToLower(msg.Role) == "assistant" {
			r = "Assistant: "
			isAssistant = true
		}
		sb.WriteString("\n")
		sb.WriteString(r)
		sb.WriteString(msg.Content)
		if isAssistant {
			sb.WriteString("\n")
		}
	}
	if strings.ToLower(req.Messages[len(req.Messages)-1].Role) != "assistant" {
		sb.WriteString("\nAssistant:")
	}
	ret := C.gsAddTask(C.uint(req.ID), C.CString(req.System), C.CString(sb.String()), C.int(req.MaxNewTokens), 0, 100)
	if ret != 0 {
		log.Println("gsAddTask failed: ", ret)
		return
	}
	log.Println("gsAddTask success: ", req.ID)
	currentTasks[req.ID] = req
}

func (task *Task) Flush() {
	if len(task.tmpBuf) <= 0 {
		return
	}
	msgBytes := make([]byte, 4+len(task.tmpBuf))
	binary.BigEndian.PutUint32(msgBytes, task.ID)
	copy(msgBytes[4:], task.tmpBuf)
	respChan <- msgBytes
	task.tmpBuf = task.tmpBuf[:0]
}

func (task *Task) AddBytes(bytes []byte) {
	if task.tmpBuf == nil {
		task.tmpBuf = make([]byte, 0, 1024)
	}
	task.tmpBuf = append(task.tmpBuf, bytes...)
	if len(task.tmpBuf) > 30 {
		if utf8.Valid(task.tmpBuf) {
			task.Flush()
		}
	}

}

func FinishTask(id uint32) {
	log.Println("[gs] FinishTask: ", id)
	task := currentTasks[id]
	if task != nil {
		task.Flush()
	}
	idBytes := make([]byte, 4)
	binary.BigEndian.PutUint32(idBytes, id)
	respChan <- idBytes

}

//export GsTokenCallback
func GsTokenCallback(id uint32, resp *C.char) {
	respBytes := []byte(C.GoString(resp))
	task := currentTasks[id]
	if task == nil {
		log.Println("[gs] unknown task id: ", id)
		return
	}

	task.AddBytes(respBytes)

}

//export GsFinishCallback
func GsFinishCallback(id uint32) {
	FinishTask(id)
}

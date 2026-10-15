//app.js
App({
  onLaunch: function () {
    //调用API从本地缓存中获取数据
    var logs = wx.getStorageSync('logs') || []
    logs.unshift(Date.now())
    wx.setStorageSync('logs', logs)
    //this.connectSoket('wss://localhost:7891')
  },

//   // 连接Socket服务端
//   connectSoket(server_url){
//     wx.connectSocket({
//       url: server_url,
//     })
//     // wx.onSocketOpen((result) => {
//     //     console.log('WebSocket 成功连接', result)
//     //     wx.onSocketMessage((result) => {
//     //         console.log('收到服务器内容：', result)
//     //     })
//     // })
//     // wx.onSocketError((err) => {
//     //     console.log('websocket连接失败', err);
//     //   })
//     this.openSocket()
// },
// //打开socket通道
// openSocket() {
//     wx.onSocketOpen(() => {
//       console.log('WebSocket 已连接')
//       this.setData({socketStatus:true})
//     })
//     wx.onSocketClose(() => {
//       console.log('WebSocket 已断开')
//       this.setData({socketStatus:false})
//     })
//     wx.onSocketError(error => {
//       console.error('发生错误', JSON.stringify(error))
//       // this.setData({
//       //   loading: false
//       // })
//     })
//     // 监听服务器推送消息
//     wx.onSocketMessage(message => {
//       console.log('收到服务器信道消息');
//     })
//   },
// // 发送游戏数据
// socket_send(){
//     let msg={
//         status: this.data.modalHidden,
//         score: this.data.score,
//         numbers: this.data.numbers,
//        }
//     wx.sendSocketMessage({
//       data: JSON.stringify(msg),
//       success:(res)=>{
//         console.log("消息发送成功", res)
//         },
//         fail(res) {
//         console.log("消息发送失败", res)
//         }
//     })
// },
})
    // 连接Socket服务端
    function connectSoket(server_url){
        wx.connectSocket({
          url: server_url
        })
        this.openSocket()
    }
    //打开socket通道
    function openSocket() {
        wx.onSocketOpen(() => {
          console.log('WebSocket 已连接')
          this.setData({socketStatus:true})
        })
        wx.onSocketClose(() => {
          console.log('WebSocket 已断开')
          this.setData({socketStatus:false})
        })
        wx.onSocketError(error => {
          console.error('发生错误', JSON.stringify(error))
          // this.setData({
          //   loading: false
          // })
        })
        // 监听服务器推送消息，并对消息做出回应
        wx.onSocketMessage(message => {
            // 收到服务器传来的action时执行动作，并返回下一次的观测状态
            var msg = JSON.parse(message.data);
            var ssx = msg['action'];
            var restart=msg['restart'];
            console.log(ssx);
            if (restart == true ) {
                // 改变状态并发送新状态
                this.modalChange();
                this.socket_send();
                }
            if (ssx == 0) {
                var reward = this.mergeAll('left');
                this.randInsert();
                this.socket_send(reward);
            }else if (ssx == 1){
                var reward = this.mergeAll('right');
                this.randInsert();
                this.socket_send(reward);
            }else if (ssx == 2){
                var reward = this.mergeAll('top');
                this.randInsert();
                this.socket_send(reward);
            }else if (ssx == 3){
                var reward = this.mergeAll('bottom');
                this.randInsert();
                this.socket_send(reward);
            }
        })
      }
    // 发送游戏数据
    function socket_send(reward){
        let msg={
            done: this.data.modalHidden,
            reward: reward,
            obs: this.data.numbers,
            send_time: Date.now(),
           }
        wx.sendSocketMessage({
          data: JSON.stringify(msg),
          success:(res)=>{
            console.log("消息发送成功", res)
            },
            fail(res) {
            console.log("消息发送失败", res)
            }
        })
    }
export{connectSoket}
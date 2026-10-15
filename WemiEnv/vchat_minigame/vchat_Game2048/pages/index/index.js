//index.js
//获取应用实例
var app = getApp()
Page({
  data: {
    score: 0,
    maxscore: 0,
    startx: 0,
    starty: 0,
    endx:0,
    endy:0,
    direction:'',
    numbers:[[0,0,2,2],[0,2,4,0],[0,4,0,0],[0,0,0,0]],
    modalHidden: true,
    socketStatus: false, //当前socket连接状态
  },
  onLoad: function () {
      var _this=this

    //调用API从本地缓存中获取数据
    var maxscore = wx.getStorageSync('maxscore')
    //连接socket服务端
    this.connectSoket('ws://localhost:7891')
    // 连接虚拟机服务器用于接入spinningup
    // this.connectSoket('ws://192.168.43.12:7891')
    // this.connectSoket('ws://192.168.136.128:7891')

    
    if(!maxscore) maxscore = 0
    this.setData({
      maxscore:maxscore
      })
      // var flask = setInterval(this.communication, 1000)
      // 以300ms为周期向客户端发送消息
    //   var test = setInterval(()=>{
    //     this.socket_send()
    //   }, 300)
    },

    // 连接Socket服务端
    connectSoket(server_url){
        wx.connectSocket({
          url: server_url
        })
        this.openSocket()
    },
    //打开socket通道
    openSocket() {
        wx.onSocketOpen(() => {
          console.log('WebSocket is connected.')
          this.setData({socketStatus:true})
        })
        wx.onSocketClose(() => {
          console.log('WebSocket is not connected.')
          this.setData({socketStatus:false})
        })
        wx.onSocketError(error => {
          console.error('Error', JSON.stringify(error))
          // this.setData({
          //   loading: false
          // })
        })
        var msg;
        var ssx;
        var restart;
        var reward;
        // 监听服务器推送消息
        wx.onSocketMessage(message => {
            // 收到服务器传来的action时执行动作，并返回下一次的观测状态
            msg = JSON.parse(message.data);
            ssx = msg['action'];
            restart=msg['restart'];
            // console.log(ssx);
            if (restart == true ) {
                // 改变状态并发送新状态
                this.modalChange();
                this.socket_send(-10);
                }
            if (ssx == 0) {
                reward = this.mergeAll('left');
                this.randInsert();
                this.socket_send(reward);
            }else if (ssx == 1){
                reward = this.mergeAll('right');
                this.randInsert();
                this.socket_send(reward);
            }else if (ssx == 2){
                reward = this.mergeAll('top');
                this.randInsert();
                this.socket_send(reward);
            }else if (ssx == 3){
                reward = this.mergeAll('bottom');
                this.randInsert();
                this.socket_send(reward);
            }
        })
      },
    // 发送游戏数据
    socket_send(reward){
        var done = !this.data.modalHidden
        let msg={
            done: done,
            reward: reward,
            obs: this.data.numbers,
            send_time: Date.now(),
           }
        wx.sendSocketMessage({
          data: JSON.stringify(msg),
          success:(res)=>{
            // console.log("消息发送成功", res)
            },
            fail(res) {
            console.log("Fail to send.", res)
            }
        })
    },

    onPullDownRefresh:function(){
        this.mergebottom();
        wx.stopPullDownRefresh();
    },
    communication:function() {
        console.log(this.data.score)
        this.send(this.data.score);
        this.receive();
    },
    storeScore:function(){
      // console.log(this.data.maxscore, this.data.score)
      if(this.data.maxscore < this.data.score){
      this.setData({
        maxscore:this.data.score
        })
        wx.setStorageSync('maxscore', this.data.maxscore)
      }
  },
  tapStart: function(event){
    this.setData({
      startx: event.touches[0].pageX,
      starty: event.touches[0].pageY
      })
  },
  tapMove: function(event){
    this.setData({
      endx: event.touches[0].pageX,
      endy: event.touches[0].pageY
      })
  },
  tapEnd: function(event){
    var heng = (this.data.endx) ? (this.data.endx - this.data.startx) : 0;
    var shu = (this.data.endy) ? (this.data.endy - this.data.starty) : 0;
    console.log(heng, shu);
    if(Math.abs(heng) > 5 || Math.abs(shu) > 5){
      var direction = (Math.abs(heng) > Math.abs(shu)) ? this.computeDir(1, heng):this.computeDir(0, shu);
      this.setData({
        startx:0,
        starty:0,
        endx:0,
        endy:0
      })
      this.mergeAll(direction);
      this.randInsert();
    }
  },
  computeDir: function(heng, num){
    if(heng) return (num > 0) ? 'right' : 'left';
    return (num > 0) ? 'bottom' : 'top';
  },
  
  send : function (score) {//向微信端发送数据

    wx.request({
        url: 'http://127.0.0.1:5000/wechat',
        data: {
            // filePath: JSON.stringify(game.filePath), //将数据格式转为JSON
            // filePath: JSON.stringify(1),
            status: JSON.stringify(this.data.modalHidden),
            score: JSON.stringify(score),
            // // 添加了通过json传参数，作为obs传入算法
            numbers: JSON.stringify(this.data.numbers),
            time: JSON.stringify(Date.parse(new Date())),
            // plains: JSON.stringify(game.plains),
            // player: JSON.stringify(game.player),
            // config: JSON.stringify(config),
        },
        method: "POST",
        header: {
            'content-type': 'application/x-www-form-urlencoded'
        },
        success: function (res) {
            //console.log(res.data);
            console.log(res.data);
        },
        fail: function (err) {
            console.log(err)
        }
    });
},

 receive : function () {
     var _this=this;
    wx.request({
        url: 'http://127.0.0.1:5000/python',
        data: {
            //action: JSON.stringify(-1), //将数据格式转为JSON
            //restart: JSON.stringify(false)
        },
        method: "POST",
        header: {
            'content-type': 'application/x-www-form-urlencoded'
        },
        success: function (res) {
            // console.log(res.data.action);
            
            var ssx = res.data.action;
            // console.log(res.data['action']);
            // getApp().globalData.restart = res.data.restart;
            var restart=res.data.restart;
            // if (getApp().globalData.restart == true ) {
            //     this.checkGame();
            // }
            if (restart == true ) {
                _this.modalChange();
                }
            if (ssx == 1) {
                _this.mergeAll('left');
                _this.randInsert();
            }else if (ssx == 2){
                _this.mergeAll('right');
                _this.randInsert();
            }else if (ssx == 3){
                _this.mergeAll('top');
                _this.randInsert();
            }else if (ssx == 4){
                _this.mergeAll('bottom');
                _this.randInsert();
            }
            // if (game.action!=0 && game.action!=undefined){
            // }
        },
        fail: function (err) {
            console.log(err)
        }
    });
},
 mergeAll: function(dir){
    this.checkGame();
    switch(dir){
        case 'left':
            return this.mergeleft();
            break;
        case 'right':
            return this.mergeright();
            break;
        case 'top':
            return this.mergetop();
            break;
        case 'bottom':
            return this.mergebottom();
            break;
        default:
        }
    },

  //左划
  mergeleft: function(){
    var change = false;
    var reward = 0;
    var arr = this.data.numbers;
    
    for(var i = 0; i < 4; i++){
      //merge first
      for(var j = 0; j < 3; j++){
        if(arr[i][j] == 0) continue;
        for(var k = 1; k < 4-j; k++){
          if(arr[i][j] != 0 && arr[i][j+k] != 0){
            if(arr[i][j] != arr[i][j+k]) break;   //不相同则直接跳过
            arr[i][j] = arr[i][j] *2;
            arr[i][j+k] = 0;
            change = true;
            reward += arr[i][j];
            // this.setData({
            // score: this.data.score + arr[i][j]/2
            // })
            if(this.data.score < arr[i][j]){
              this.setData({score:arr[i][j]})
            }
            break;
          }
        }
      }
      //movemove
      for(var j = 0; j < 3; j++){
        if(arr[i][j] == 0){
          for(var k = 1; k < 4-j; k++){
            if(arr[i][j+k] != 0){
              arr[i][j] = arr[i][j+k];
              arr[i][j+k] = 0;
              change = true;
              break;
            }
          }
        }
      }
    }
    this.setData({
          numbers:arr
          })
    this.storeScore()
    return reward;
  },
  //右滑
  mergeright: function(){
    var change = false
    var reward = 0;
    var arr = this.data.numbers;
    
    for(var i = 0; i < 4; i++){
      //merge first
      for(var j = 3; j > 0; j--){
        if(arr[i][j] == 0) continue;
        for(var k = 1; k <= j; k++){
          if(arr[i][j] != 0 && arr[i][j-k] != 0){
            if(arr[i][j] != arr[i][j-k]) break;
            arr[i][j] = arr[i][j] *2;
            arr[i][j-k] = 0;
            change = true;
            reward += arr[i][j];
            // this.setData({
            // score: this.data.score + arr[i][j]/2
            // })
            if(this.data.score < arr[i][j]){
              this.setData({score:arr[i][j]})
            }
            break;
          }
        }
      }
      //movemove
      for(var j = 3; j > 0; j--){
        if(arr[i][j] == 0){
          for(var k = 1; k <= j; k++){
            if(arr[i][j-k] != 0){
              arr[i][j] = arr[i][j-k];
              arr[i][j-k] = 0;
              change = true;
              break;
            }
          }
        }
      }
    }
    this.setData({
          numbers:arr
          })
    this.storeScore()
    return reward;
  },
  //下划
  mergebottom: function(){
    var change = false
    var reward = 0;
    var arr = this.data.numbers;
    
    for(var i = 0; i < 4; i++){
      //merge first
      for(var j = 3; j > 0; j--){
        if(arr[j][i] == 0) continue;
        for(var k = 1; k <= j; k++){
          if(arr[j][i] != 0 && arr[j-k][i] != 0){
            if(arr[j][i] != arr[j-k][i]) break;
            arr[j][i] = arr[j][i] *2;
            arr[j-k][i] = 0;
            change = true
            reward += arr[j][i];
            // this.setData({
            // score: this.data.score + arr[j][i]/2
            // })
            if(this.data.score < arr[j][i]){
              this.setData({score:arr[j][i]})
            }
            break;
          }
        }
      }
      //movemove
      for(var j = 3; j > 0; j--){
        if(arr[j][i] == 0){
          for(var k = 1; k <= j; k++){
            if(arr[j-k][i] != 0){
              arr[j][i] = arr[j-k][i];
              arr[j-k][i] = 0;
              change = true
              break;
            }
          }
        }
      }
    }
    this.setData({
          numbers:arr
          })
    this.storeScore()
    return reward
  },
  //上滑
  mergetop: function(){
    var change = false
    var reward = 0;
    var arr = this.data.numbers;
    
    for(var i = 0; i < 4; i++){
      //merge first
      for(var j = 0; j < 3; j++){
        if(arr[j][i] == 0) continue;
        for(var k = 1; k < 4-j; k++){
          if(arr[j][i] != 0 && arr[j+k][i] != 0){
            if(arr[j][i] != arr[j+k][i]) break;
            arr[j][i] = arr[j][i] *2;
            arr[j+k][i] = 0;
            change = true
            reward += arr[j][i];
            // this.setData({
            // score: this.data.score + arr[j][i]/2
            // })
            if(this.data.score < arr[j][i]){
              this.setData({score:arr[j][i]})
            }
            break;
          }
        }
      }
      //movemove
      for(var j = 0; j < 3; j++){
        if(arr[j][i] == 0){
          for(var k = 1; k < 4-j; k++){
            if(arr[j+k][i] != 0){
              arr[j][i] = arr[j+k][i];
              arr[j+k][i] = 0;
              change = true
              break;
            }
          }
        }
      }
    }
    this.setData({
          numbers:arr
          })
    this.storeScore()
    return reward
  },
  //自定义种子随机
  rnd: function( seed ){
    seed = ( seed * 9301 + 49297 ) % 233280; 
    return seed / ( 233280.0 );
},

  //随机插入
  randInsert: function(){
    var arr = this.data.numbers
    //随机2或4
    // var num = Math.random() < 0.8 ? 2 : 4
    var num = this.rnd(1) < 0.8 ? 2 : 4
    //计算随机位置
    var zeros = [];
    for(var i = 0; i < 4; i++){
      for(var j = 0; j < 4; j++){
        if(arr[i][j] == 0){
            zeros.push([i, j]);
        }
      }
    }
    if(zeros.length > 0){
        // var position = zeros[Math.floor(Math.random() * zeros.length)];
        var position = zeros[Math.floor(this.rnd(1) * zeros.length)];
        arr[position[0]][position[1]] = num
    }
    this.setData({
      numbers:arr
      })
    // 检查游戏是否结束
    this.checkGame()
  },
  checkGame: function(){
    var arr = this.data.numbers
    for(var i = 0; i < 4; i++){
      for(var j = 0; j < 4; j++){
        if(arr[i][j] == 0) return;
      }
    }
    for(var i = 0; i < 3; i++){
      for(var j = 0; j < 3; j++){
        if(arr[i][j] == arr[i+1][j] || arr[i][j] == arr[i][j+1]) return;
      }
    }
        
    for(var j = 0; j < 3; j++){
      if(arr[3][j] == arr[3][j+1]) return;
      if(arr[j][3] == arr[j+1][3]) return;
    }
    this.setData({
      modalHidden: false,
    })
  },
  modalChange:function(){
    this.setData({
      score: 0,
      numbers:[[0,0,2,2],[0,2,4,0],[0,4,0,0],[0,0,0,0]],
      modalHidden: true,
    })
  },
  modalCancle:function(){
    this.setData({
      modalHidden: true,
    })
  },
})

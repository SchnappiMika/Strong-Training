const emitter = require('../utils/emitter.js');

function wxBird(opts,page) {
    var ctx = this.ctx = opts.ctx;
    var id = this.id = opts.id
    var cwidth = this.cwidth = opts.width;
    var cheight = this.cheight = opts.height;
    var objects = this.objects = [];
    var birdIndex = this.birdIndex = 0;

    // console.log("path");
    // console.log(res.tempFilePath);

    var ver1 = this.ver1 = 10;
    var stoped= this.stoped = false;
    var fire = this.__proto__.fire;
    var ver2 = this.ver2;
    var gravity = this.gravity = 2;
    var pipe_height = this.pipe_height = 200;
    var velocity = this.velocity = 10;
    var drawTimer = this.drawTimer = 0;
    var drawTimer2 = this.drawTimer2 = 0;
    var score = this.score = 0;
    var reward = this.reward = 0.1;
    var pubnish = this.pubnish = 0;

    var isScore = this.isScore = false;
    // console.log("opts");
    // console.log(opts);
    // console.log("this");
    // console.log(this);
    // console.log(ctx);
    var birds = this.birds = ["../../images/0.gif", "../../images/1.gif", "../../images/2.gif"];

    var back = this.back = new Background(ctx, 0, 0, 400, 600, "../../images/bg.png");
    var up_pipe = this.up_pipe = new UpPipe(ctx, 0, 0, 50, 200, "../../images/pipe.png");
    var down_pipe = this.down_pipe = new DownPipe(ctx, 0, 400, 50, 200, "../../images/pipe.png");
    var ground = this.ground = new Background(ctx, 0, 550, 400, 200, "../../images/ground.png");
    var bird = this.bird = new Bird(ctx, 80, 300, 40, 40, birds);

    objects.push(back);
    objects.push(up_pipe);
    objects.push(down_pipe);
    objects.push(ground);
    objects.push(bird);
    
    function UpPipe(ctx, x, y, width, height, img_src) {
        this.ctx = ctx;
        this.px = x;
        this.py = y;
        this.pwidth = width;
        this.pheight = height;
        this.img_src = img_src;
        this.draw = drawUpPipe;
    }
    
    function drawUpPipe() {
        this.ctx.drawImage(this.img_src, 150, 500, 150, 800, this.px, this.py, this.pwidth, this.pheight);
    }

    function DownPipe(ctx, x, y, width, height, img_src) {
        this.ctx = ctx;
        this.px = x;
        this.py = y;
        this.pwidth = width;
        this.pheight = height;
        this.img_src = img_src;
        this.draw = drawDownPipe;
    }

    function drawDownPipe() {
        this.ctx.drawImage(this.img_src, 0, 500, 150, 500, this.px, this.py, this.pwidth, this.pheight);
    }

    function Background(ctx, x, y, width, height, img_src) {
        this.ctx = ctx;
        this.bgx = x;
        this.bgy = y;
        this.bgwidth = width;
        this.bgheight = height;
        this.img_src = img_src;
        this.draw = drawbg;
    }

    function drawbg() {
        this.ctx.drawImage(this.img_src, this.bgx, this.bgy, this.bgwidth, this.bgheight);
    }

    function Bird(ctx, x, y, width, height, img_srcs) {
        this.ctx = ctx;
        this.bx = x;
        this.by = y;
        this.bwidth = width;
        this.bheight = height;
        this.imgs = img_srcs;
        this.draw = drawbird;
    }

    function drawbird() {
        birdIndex++;
        this.img_src = this.imgs[birdIndex % 3];
        this.ctx.drawImage(this.img_src, this.bx, this.by, this.bwidth, this.bheight);
    }
    var timer;
    var game = this.game = {};
    game.action;
    game.count;
    game.stoped = this.stoped;
    game.birdIndex = this.birdIndex ;
    game.ver1 = this.ver1 ;
    game.ver2 = this.ver2;
    game.gravity = this.gravity;
    game.pipe_height = this.pipe_height;
    game.velocity = this.velocity;
    game.drawTimer = this.drawTimer;
    game.drawTimer2 = this.drawTimer2;
    game.pubnish = this.pubnish;
    game.score = this.score;
    game.isScore = this.isScore;
    game.bird = this.bird;
    game.up_pipe = this.up_pipe;
    game.down_pipe = this.down_pipe;
    getApp().globalData.restart=false;
    var that = this;

    // 通讯建立部分
    connectSoket('ws://localhost:7891');
    // connectSoket('ws://192.168.43.12:7891');
    // connectSoket('ws://192.168.136.128:7891');
    // 连接Socket服务端
    function connectSoket(server_url){
      wx.connectSocket({
        url: server_url
      })
      openSocket()
  }
  //打开socket通道
  function openSocket() {
      wx.onSocketOpen(() => {
        console.log('WebSocket is connected.')
      })
      wx.onSocketClose(() => {
        console.log('WebSocket is not connected.')
      })
      wx.onSocketError(error => {
        console.error('Error', JSON.stringify(error))
      })
      // 监听服务器推送消息，并对消息做出回应
      wx.onSocketMessage(message => {
          // 收到服务器传来的action时执行动作，并返回下一次的观测状态
          // console.log(message);
          var msg = JSON.parse(message.data);
          var ssx = msg['action'];
          var restart=msg['restart'];
          // console.log(ssx);
          if (restart == true ) {
              // 改变状态并发送新状态
              clearInterval(timer);
              game.startGame1();
              // 过300ms再发送新状态
              setTimeout(() => {socket_send();}, 100);
              }
          if (ssx == 0) {
            // 防止交互过快
            setTimeout(() => {socket_send();}, 100);
          }else if (ssx == 1){
            keyup();
            // 防止交互过快
            // 过300ms再发送新状态
            setTimeout(() => {socket_send();}, 100);
          }
      })
    }
  // 发送游戏数据
  function socket_send(){
    var obs=[bird.by, down_pipe.py, up_pipe.pheight];
    var reward;
    if(stoped == true) reward = -10;
    else if(bird.by > up_pipe.pheight && bird.by < down_pipe.py) {
      if(bird.by - up_pipe.pheight > down_pipe.py - bird.by) reward = down_pipe.py - bird.by;
      else reward = bird.by - up_pipe.pheight;
    }
    else if(bird.by < up_pipe.pheight) reward = bird.by - up_pipe.pheight;
    else if(bird.by > down_pipe.py) reward = down_pipe.py - bird.by;
      let msg={
        // 游戏观测状态
        obs: obs,
        // 游戏结束信息
        done: stoped,
        // 奖励
        reward: reward,
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
  }

    this.game.startGame1 = function() {
        //判断游戏是否结束
        stoped = false;
        game.ver1 = 10;
        gravity = 2;
        pipe_height = 200;
        velocity = 10;
        score = 0;
        reward = 0.1;
        // 失败的惩罚
        pubnish = 0;
        isScore = false;
        // 小鸟和管道状态信息
        bird.by = 300;
        up_pipe.px = 0;
        up_pipe.pheight = 200;
        down_pipe.px = 0;
        down_pipe.py = 400;
        down_pipe.pheight = 200;
        // game.drawall1();
        timer = setInterval(game.drawall1, 40);
    }
    this.game.drawall1 = function() {
        if (stoped) return
        ctx.clearRect(0, 0, cwidth, cheight);
        var i;
        for (i = 0; i < objects.length; i++) {
            objects[i].draw();
        }
        
        game.calculator1();
    
        wx.drawCanvas({
            canvasId: id,
            actions: ctx.getActions()
        })
        // console.log(velocity, " refresh");
        // drawTimer = setTimeout(() => {
        //     game.drawall1()
        // }, 40)
    }
    
    function keyup(){
      bird.by -= 60;
      if (bird.by < 0)
        bird.by = 0
    }

    this.game.calculator1 = function(){
        // 判断游戏是否结束
        if (bird.by + bird.bheight > ground.bgy ||
            ((bird.bx + bird.bwidth > up_pipe.px) && (bird.by > up_pipe.py) && (bird.bx + bird.bwidth < up_pipe.px + up_pipe.pwidth) && (bird.by < up_pipe.py + up_pipe.pheight)) ||
            ((bird.bx + bird.bwidth > up_pipe.px) && (bird.by > up_pipe.py) && (bird.bx + bird.bwidth < up_pipe.px + up_pipe.pwidth) && (bird.by < up_pipe.py + up_pipe.pheight)) ||
            ((bird.bx > down_pipe.px) && (bird.by > down_pipe.py) && (bird.bx < down_pipe.px + down_pipe.pwidth) && (bird.by < down_pipe.py + down_pipe.pheight)) ||
            ((bird.bx > down_pipe.px) && (bird.by + bird.bheight > down_pipe.py) && (bird.bx < down_pipe.px + down_pipe.pwidth) && (bird.by + bird.bheight < down_pipe.py + down_pipe.pheight))) {
            stoped = true;
            pubnish = -50;
            clearTimeout(drawTimer)
            ctx.setFillStyle("rgb(255,255,255)");
            ctx.setFontSize(30);
            ctx.fillText("您挂了,得分：" + score + "!", 90, 100)
            //弹出游戏对话框
            //fire('over', { score: score });
            // score =-1;
            return;
        }
        // 小鸟下降的过程
        ver2 = ver1 + gravity;
        bird.by += (ver2 + ver1) * 0.25;
    
        if (up_pipe.px + up_pipe.pwidth > 0) {
            // 表示小鸟水平移动的过程
            up_pipe.px -= velocity;
            down_pipe.px -= velocity;
        } else {
            // 随机生成新的管道
            up_pipe.px = 400;
            down_pipe.px = 400;
            up_pipe.pheight = 100 + Math.random() * 200;
            down_pipe.py = up_pipe.pheight + pipe_height;
            down_pipe.pheight = 600 - down_pipe.py;
            isScore = true;
        }
        // 飞过一组管道 分数++ 速度++
        if (isScore && bird.bx > up_pipe.px + up_pipe.pwidth) {
            score += 1;
            // reward = 1;
            isScore = false;
            // 暂时不改变速度
            // if (score > 0 && score % 10 === 0) {
            //     velocity++;
            // }
        //   console.log(reward)
    
        }
    
        ctx.setFillStyle("rgb(255,255,255)");
        ctx.setFontSize(30);
        if (score > 0) {
            score % 10 !== 0 ? ctx.fillText(score, 180, 100) : ctx.fillText("非常棒!" + score, 120, 100);
        }
    }
    this.game.keyup1 = function(){
                bird.by -= 80;
                if (bird.by < 0)
                    bird.by = 0
    }
}

wxBird.prototype.calculator = function () {
    this.game.calculator1()
}

wxBird.prototype.drawall = function () {
    this.game.drawall1()
}

wxBird.prototype.startGame = function () {
    this.game.startGame1()
}
wxBird.prototype.keyup = function () {
    this.bird.by -= 60;
    if (this.bird.by < 0)
        this.bird.by = 0
}

emitter.setup(wxBird.prototype);

module.exports = wxBird;
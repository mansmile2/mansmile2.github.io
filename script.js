const startBtn = document.querySelector("#start");
const screens = document.querySelectorAll(".screen");
const timeList = document.querySelector("#time-list");
const timeEl = document.querySelector("#time");
const board = document.querySelector("#board");
const colors = ["#3cc","#e6ffff","#ffe6e6","#fcc","#c00","#933"];

let time = 0;
let score = 0;
let highScore = 0;
let timeIntervalId;

startBtn.addEventListener("click", (event) => {
  event.preventDefault();
  screens[0].classList.add("up");
});

timeList.addEventListener("click", (event) => {
  if (event.target.classList.contains("time-btn")) {
    time = parseInt(event.target.getAttribute("data-time"));
    screens[1].classList.add("up");
    startGame();
  }
});

board.addEventListener("click", (event) => {
  if (event.target.classList.contains("circle")) {
    score++;
    event.target.remove();
    createRandomCircle();
  }
});

function startGame() {
  board.innerHTML = "";
  if(timeIntervalId) {
    clearInterval(timeIntervalId);
  }
  timeIntervalId = setInterval(decreaseTime, 1000);
  createRandomCircle();
  setTime(time);
}

function decreaseTime() {
  if (time === 0) {
    checkHighScore(score);
    finishGame();
  } else {
    let current = --time;
    if (current < 10) {
      current = `0${current}`;
    }
    setTime(current);
  }
}

function setTime(value) {
  timeEl.innerHTML = `00:${value}`;
}


function finishGame() {
  timeEl.innerHTML = `<h2>Рекорд: <span class="primary">${highScore}</span></h2>`;
  board.innerHTML = `<h1>Счет: <span class="primary">${score}</span></h1>`;
  const startAgainBtn = document.createElement("a");
  startAgainBtn.setAttribute("href", "#");
  startAgainBtn.classList.add("start");
  startAgainBtn.setAttribute("id", "start-again");
  startAgainBtn.innerText = "Играть заново";
  board.append(startAgainBtn);
  const startAgainBtnEl = document.querySelector("#start-again");
  const restartGame = () => {
    time = 0;
    score = 0;
    board.classList.remove("hide");
    screens[0].classList.remove("up");
    screens[1].classList.remove("up");
    timeEl.parentNode.classList.remove("hide");
    startGame();
  };
  startAgainBtnEl.addEventListener("click", restartGame);
}

function checkHighScore(score) {
  if (score > highScore) {
    highScore = score;
  }
}

function createRandomCircle() {
  const circle = document.createElement("div");
  const size = getRandomNumber(10, 60);
  const color = getRandomColor();
  const { width, height } = board.getBoundingClientRect();

  const x = getRandomNumber(0, width - size);
  const y = getRandomNumber(0, height - size);

  circle.classList.add("circle");
  circle.style.width = `${size}px`;
  circle.style.height = `${size}px`;
  circle.style.top = `${y}px`;
  circle.style.left = `${x}px`;
  circle.style.background = `linear-gradient(90deg, ${color} 0%, ${color} 100%)`;

  board.append(circle);
}

function getRandomNumber(min, max) {
  return Math.round(Math.random() * (max - min) + min);
}

function getRandomColor() {
  return colors[Math.floor(Math.random() * colors.length)];
}

function setColor(event) {
  const element = event.target;
  const color = getRandomColor();
  element.style.backgroundColor = color;
  element.style.boxShadow = `0 0 2px ${color}, 0 0 20px ${color}`;
}

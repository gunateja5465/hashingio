var nav=document.querySelector('.navbar')
var clicked=false
var span=document.querySelector('span')
function clickme(){
	clicked=!clicked
	if(clicked){
		document.querySelector('.ul').style.display='flex'
		aa=document.querySelectorAll('.ul a')
		aa.forEach((mm)=>{
			mm.addEventListener('click',()=>{
				document.querySelector('.ul').style.display='none'
				span.innerHTML="#"
				clicked=!clicked
			})
		})
		span.innerHTML="!!"
	}
	else{
		document.querySelector('.ul').style.display='none'
		span.innerHTML="#"
	}
}
var a=document.querySelectorAll('a')
a.forEach((i)=>{i.addEventListener('click',()=>{
    a.forEach((j)=>{j.classList.remove('active')})
    i.classList.toggle('active')
})})
var aa;
window.onscroll=()=>{
var y=window.pageYOffset
if(y>1){
    nav.classList.add('navbartoggle')
    var a=document.querySelectorAll('a')
    a.forEach((i)=>{i.style.color="grey"})
}
else{
    nav.classList.remove('navbartoggle')
    var a=document.querySelectorAll('a')
    a.forEach((i)=>{i.style.color=""})
}
}

// var span=document.querySelector('span')
// span.addEventListener('click',()=>{
// 	clicked=!clicked
// 	if(clicked){
// 		document.querySelector('.ul').style.display='flex'
// 		aa=document.querySelectorAll('.ul a')
// 		span.innerHTML="!!"
// 	}
// 	else{
// 		document.querySelector('.ul').style.display='none'
// 		span.innerHTML="#"
// 	}
// })

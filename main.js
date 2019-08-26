var btn = document.getElementById("SEND");

btn.addEventListener("onclick", function(){
    ourRequest = new XMLHttpRequest();
    ourRequest.open('GET', 'bot.php');
    ourRequest.onload = function(){
        var resultat = JSON.parse(ourRequest.responseText);
        var html = resultat.map(function(message){
           return   
        })
    }
})
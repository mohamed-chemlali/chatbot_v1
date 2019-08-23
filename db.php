<?php
$servername="localhost";
$username="root";
$password="";
$dbname="chatbot";

$conn=new mysqli($servername,$username,$password,$dbname);

if($conn){
    echo "connected";   
}
else{
    echo "connection failed";
}

?>
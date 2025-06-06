<?php
// 启动会话
session_start();
include 'config.php';
include 'includes/language-loader.php';
include 'includes/log-functions.php';

// 记录登出操作
$username = isset($_SESSION['username']) ? $_SESSION['username'] : 'Unknown user';
$user_id = isset($_SESSION['id']) ? $_SESSION['id'] : null;

// 添加登出激活日志
log_info("Logout operation initiated: $username", "logout.php", $user_id);

// 添加登出完成日志
log_info("User logged out: $username", "logout.php", $user_id);

// 清除所有会话变量
$_SESSION = array();

// 如果使用了会话cookie，则清除会话cookie
if (isset($_COOKIE[session_name()])) {
    setcookie(session_name(), '', time() - 42000, '/');
}

// 销毁会话
session_destroy();
?>
<!DOCTYPE html>
<html>
<head>
    <title>登出 - FRAS 系统</title>
    <script>
        // 清除会话存储中的标记
        sessionStorage.removeItem('loginLoaded');
        sessionStorage.removeItem('dashboardLoaded');
        sessionStorage.removeItem('fromDashboard');

        // 重定向到登录页面
        window.location.href = 'login.php';
    </script>
</head>
<body>
    <p>正在登出...</p>
</body>
</html>
<?php
// 如果 JavaScript 未执行，则使用 PHP 重定向
exit();
?>

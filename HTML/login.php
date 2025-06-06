<?php
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

// 设置缓存控制头，防止浏览器缓存
header("Cache-Control: no-store, no-cache, must-revalidate, max-age=0");
header("Cache-Control: post-check=0, pre-check=0", false);
header("Pragma: no-cache");

session_start();
include 'config.php';
include 'includes/log-functions.php';
include 'includes/notification-functions.php';
include 'includes/language-loader.php';

$error = "";

// 在session中记录登录尝试次数
if (!isset($_SESSION['login_attempts'])) {
    $_SESSION['login_attempts'] = 0;
    $_SESSION['last_attempt_time'] = time();
}

// 如果尝试次数过多，暂时锁定账户
if ($_SESSION['login_attempts'] >= 5) {
    $time_passed = time() - $_SESSION['last_attempt_time'];
    if ($time_passed < 300) { // 5分钟锁定期
        $error = "Too many failed login attempts. Please try again after " . ceil((300 - $time_passed) / 60) . " minutes.";
        $login_blocked = true; // 标记登录被阻止
    } else {
        // 重置尝试次数
        $_SESSION['login_attempts'] = 0;
        $login_blocked = false;
    }
} else {
    $login_blocked = false;
}

if ($_SERVER["REQUEST_METHOD"] == "POST" && !$login_blocked) {
    $username = $_POST['username'];
    $password = $_POST['password'];

    $sql = "SELECT * FROM users WHERE username = ?";
    $stmt = $conn->prepare($sql);

    if (!$stmt) {
        die("Prepare failed: " . $conn->error);
    }

    $stmt->bind_param("s", $username);
    $stmt->execute();
    $result = $stmt->get_result();

    if ($result->num_rows === 1) {
        $user = $result->fetch_assoc();

        if (password_verify($password, $user['password'])) {
            // 登录成功，重置登录尝试次数
            $_SESSION['login_attempts'] = 0;

            $_SESSION['loggedin'] = true;
            $_SESSION['id'] = $user['employee_id'];
            $_SESSION['username'] = $user['username'];
            $_SESSION['role'] = $user['role'];

            // 记录登录成功日志
            log_info("User logged in successfully", "login.php", $user['employee_id']);

            // 不立即跳转，而是显示动画
            $_SESSION['login_success'] = true;
        } else {
            // 密码错误，增加登录尝试次数
            $_SESSION['login_attempts']++;
            $_SESSION['last_attempt_time'] = time();
            $error = "Wrong password";

            // 记录登录失败日志
            log_warning("Failed login attempt: wrong password for user " . $username, "login.php");
        }
    } else {
        // 用户不存在，增加登录尝试次数
        $_SESSION['login_attempts']++;
        $_SESSION['last_attempt_time'] = time();
        $error = "User not found";

        // 记录登录失败日志
        log_warning("Failed login attempt: user not found - " . $username, "login.php");
    }

    $stmt->close();
}
?>
<!DOCTYPE html>
<html lang="UTF-8">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- 防止浏览器缓存 -->
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <link rel="icon" href="icon.png" type="image/x-icon">
    <title>Login - FRAS System</title>
    <link rel="stylesheet" href="css/login-style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- 引入通知系统 -->
    <link rel="stylesheet" href="css/notifications.css">
    <script src="js/notifications.js"></script>
</head>
<body>
    <div id="notification-area"></div>

    <div class="login-box" id="login-box">
        <div class="logo-container">
            <img src="icon.png" alt="FRAS Logo" class="logo" id="fras-logo">
        </div>
        <h2 class="title" id="login-title">Welcome to FRAS</h2>
        <form method="POST" action="javascript:void(0);" id="login-form">
            <input type="text" name="username" placeholder="Username" required class="input-field">
            <input type="password" name="password" placeholder="Password" required class="input-field">
            <button type="submit" class="login-button">Login</button>

            <div class="signup-text">
                <?php echo __('no_account'); ?> <a href="register.php"><?php echo __('register_here'); ?></a>
            </div>

            <?php if ($error): ?>
                <script>
                    document.addEventListener('DOMContentLoaded', function() {
                        showErrorNotification('<?php echo addslashes($error); ?>');
                    });
                </script>
            <?php endif; ?>
        </form>
    </div>

    <?php
        // 如果登录成功，直接重定向到仪表盘
        if (isset($_SESSION['login_success']) && $_SESSION['login_success']) {
            // 清除登录成功标记
            unset($_SESSION['login_success']);

            // 使用JavaScript重定向到仪表盘
            echo '<script>window.location.href = "dashboard.php";</script>';
            exit;
        }
    ?>
    <!-- 正常登录表单的脚本 -->
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            // 处理表单提交
            const loginForm = document.getElementById('login-form');

            loginForm.addEventListener('submit', function(e) {
                e.preventDefault();

                // 创建FormData对象
                const formData = new FormData(loginForm);

                // 使用fetch API发送POST请求
                fetch('login-ajax.php', {
                    method: 'POST',
                    body: formData,
                    credentials: 'same-origin'
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // 清除登录页面已加载标记，确保下次登录正常
                        sessionStorage.removeItem('loginLoaded');
                        // 清除从dashboard返回的标记
                        sessionStorage.removeItem('fromDashboard');

                        // 直接重定向到仪表盘
                        window.location.href = 'dashboard.php';
                    } else {
                        // 登录失败，显示错误
                        showErrorNotification(data.message);
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    showErrorNotification('An error occurred. Please try again.');
                });
            });

            // 使用会话存储来防止重复刷新
            const hasLoaded = sessionStorage.getItem('loginLoaded');

            // 检查是否是从dashboard返回的
            const fromDashboard = document.referrer.includes('dashboard.php');

            if (!hasLoaded || fromDashboard) {
                // 第一次加载页面或从dashboard返回时设置标记
                sessionStorage.setItem('loginLoaded', 'true');

                // 不再需要清除动画相关标记
            }

            // 处理浏览器后退按钮
            window.addEventListener('pageshow', function(event) {
                // 如果是从缓存加载的页面，刷新页面以避免缓存问题
                if (event.persisted) {
                    // 不再需要清除动画相关标记

                    // 如果需要刷新
                    if (!hasLoaded) {
                        window.location.reload();
                    }
                }
            });
        });
    </script>

</body>
</html>

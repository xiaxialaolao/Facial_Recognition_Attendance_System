<?php
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

// 设置缓存控制头，防止浏览器缓存
header("Cache-Control: no-store, no-cache, must-revalidate, max-age=0");
header("Cache-Control: post-check=0, pre-check=0", false);
header("Pragma: no-cache");

// 设置响应头为 JSON
header('Content-Type: application/json');

session_start();
include 'config.php';
include 'includes/log-functions.php';
include 'includes/notification-functions.php';
include 'includes/language-loader.php';

$response = ['success' => false, 'message' => ''];

// 在session中记录登录尝试次数
if (!isset($_SESSION['login_attempts'])) {
    $_SESSION['login_attempts'] = 0;
    $_SESSION['last_attempt_time'] = time();
}

// 如果尝试次数过多，暂时锁定账户
if ($_SESSION['login_attempts'] >= 5) {
    $time_passed = time() - $_SESSION['last_attempt_time'];
    if ($time_passed < 300) { // 5分钟锁定期
        $response['message'] = "Too many failed login attempts. Please try again after " . ceil((300 - $time_passed) / 60) . " minutes.";

        // 记录账户锁定日志
        log_warning("Account temporarily locked due to too many failed login attempts", "login-ajax.php");

        echo json_encode($response);
        exit;
    } else {
        // 重置尝试次数
        $_SESSION['login_attempts'] = 0;

        // 记录锁定期结束日志
        log_info("Account lock period ended, login attempts reset", "login-ajax.php");
    }
}

if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $username = $_POST['username'];
    $password = $_POST['password'];

    $sql = "SELECT * FROM users WHERE username = ?";
    $stmt = $conn->prepare($sql);

    if (!$stmt) {
        $response['message'] = "Database error: " . $conn->error;
        echo json_encode($response);
        exit;
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
            log_info("User logged in successfully", "login-ajax.php", $user['employee_id']);

            // 设置登录成功标记
            $_SESSION['login_success'] = true;

            $response['success'] = true;
            $response['message'] = "Login successful";
            // 不再显示动画
            $response['show_animation'] = false;
        } else {
            // 密码错误，增加登录尝试次数
            $_SESSION['login_attempts']++;
            $_SESSION['last_attempt_time'] = time();
            $response['message'] = "Wrong password";

            // 记录登录失败日志
            log_warning("Failed login attempt: wrong password for user " . $username, "login-ajax.php");
        }
    } else {
        // 用户不存在，增加登录尝试次数
        $_SESSION['login_attempts']++;
        $_SESSION['last_attempt_time'] = time();
        $response['message'] = "User not found";

        // 记录登录失败日志
        log_warning("Failed login attempt: user not found - " . $username, "login-ajax.php");
    }

    $stmt->close();
}

echo json_encode($response);

<?php
// log-ajax.php - 处理AJAX日志请求
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

// 检查用户是否已登录
if (!isset($_SESSION['loggedin']) || $_SESSION['loggedin'] !== true) {
    echo json_encode(['success' => false, 'message' => 'User not logged in']);
    exit;
}

// 获取当前登录用户信息
$current_username = $_SESSION['username'];
$current_user_id = $_SESSION['id'];

// 处理AJAX请求
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $action = $_POST['action'] ?? '';
    
    if ($action === 'log_error') {
        // 获取错误消息和来源
        $message = $_POST['message'] ?? 'Unknown error';
        $source = $_POST['source'] ?? 'log-ajax.php';
        
        // 记录错误日志
        log_error($message, $source);
        
        // 返回成功响应
        echo json_encode(['success' => true, 'message' => 'Error logged successfully']);
        exit;
    } else if ($action === 'log_info') {
        // 获取信息消息和来源
        $message = $_POST['message'] ?? 'Unknown info';
        $source = $_POST['source'] ?? 'log-ajax.php';
        
        // 记录信息日志
        log_info($message, $source);
        
        // 返回成功响应
        echo json_encode(['success' => true, 'message' => 'Info logged successfully']);
        exit;
    } else if ($action === 'log_warning') {
        // 获取警告消息和来源
        $message = $_POST['message'] ?? 'Unknown warning';
        $source = $_POST['source'] ?? 'log-ajax.php';
        
        // 记录警告日志
        log_warning($message, $source);
        
        // 返回成功响应
        echo json_encode(['success' => true, 'message' => 'Warning logged successfully']);
        exit;
    }
}

// 如果没有匹配的操作或请求方法不是POST，返回错误
echo json_encode(['success' => false, 'message' => 'Invalid request']);

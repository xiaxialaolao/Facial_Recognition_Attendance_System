<?php
// 通知系统加载器 - 在所有页面中包含

// 如果调试函数不存在，定义一个
if (!function_exists('debug_log')) {
    function debug_log($message, $data = null) {
        $log_message = date('[Y-m-d H:i:s]') . " $message";
        if ($data !== null) {
            $log_message .= ": " . print_r($data, true);
        }
        error_log($log_message);
    }
}

debug_log("notifications-loader.php 开始执行");
debug_log("当前会话状态", $_SESSION);

// 初始化通知变量
$notification_message = '';
$notification_type = 'success';

// 处理会话中的通知（优先级最高）
if (isset($_SESSION['notification'])) {
    debug_log("发现会话中的通知", $_SESSION['notification']);
    $notification = $_SESSION['notification'];
    $notification_message = isset($notification['message']) ? $notification['message'] : '';
    $notification_type = isset($notification['type']) ? $notification['type'] : 'success';

    // 清除会话中的通知，防止刷新页面时再次显示
    unset($_SESSION['notification']);
    debug_log("已清除会话中的通知");
} else {
    debug_log("会话中没有通知");
}

// 处理页面中的直接通知（优先级低于会话通知）
debug_log("检查页面变量", ['error' => isset($error) ? $error : 'not set', 'success' => isset($success) ? $success : 'not set']);

if (empty($notification_message) && isset($error) && !empty($error)) {
    debug_log("使用页面错误消息", $error);
    $notification_message = $error;
    $notification_type = 'error';
}

if (empty($notification_message) && isset($success) && !empty($success)) {
    debug_log("使用页面成功消息", $success);
    $notification_message = $success;
    $notification_type = 'success';
}

// 调试信息 - 帮助排查问题
debug_log("最终通知状态", ['message' => $notification_message, 'type' => $notification_type]);
?>

<!-- 通知系统CSS和JavaScript -->
<link rel="stylesheet" href="css/notifications.css">
<script src="js/notifications.js"></script>

<!-- 用于显示通知的JavaScript函数 -->
<script>
    document.addEventListener('DOMContentLoaded', function() {
        <?php if (!empty($notification_message)): ?>
            showNotification('<?php echo addslashes($notification_message); ?>', '<?php echo $notification_type; ?>');
        <?php endif; ?>
    });
</script>

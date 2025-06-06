<?php
/**
 * 通知函数 - 用于在PHP代码中设置通知
 */

/**
 * 设置通知消息，将在下一个页面加载时显示
 * 
 * @param string $message 通知消息
 * @param string $type 通知类型 ('success', 'error', 'warning', 'info')
 */
function set_notification($message, $type = 'success') {
    $_SESSION['notification'] = [
        'message' => $message,
        'type' => $type
    ];
}

/**
 * 设置成功通知
 * 
 * @param string $message 通知消息
 */
function set_success_notification($message) {
    set_notification($message, 'success');
}

/**
 * 设置错误通知
 * 
 * @param string $message 通知消息
 */
function set_error_notification($message) {
    set_notification($message, 'error');
}

/**
 * 设置警告通知
 * 
 * @param string $message 通知消息
 */
function set_warning_notification($message) {
    set_notification($message, 'warning');
}

/**
 * 设置信息通知
 * 
 * @param string $message 通知消息
 */
function set_info_notification($message) {
    set_notification($message, 'info');
}

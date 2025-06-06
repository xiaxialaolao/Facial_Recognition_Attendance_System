<?php
// 日期格式处理函数

// 设置固定的日期格式
$date_format = 'Y-m-d'; // 使用固定的格式 YYYY-MM-DD

// 格式化日期函数
function formatDate($date, $include_time = false) {
    global $date_format;

    if ($include_time) {
        return date($date_format . ' H:i', strtotime($date));
    } else {
        return date($date_format, strtotime($date));
    }
}
?>

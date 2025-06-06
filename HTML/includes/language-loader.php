<?php
// 语言加载器 - 在所有页面中包含

// 获取当前用户的语言设置
function getUserLanguage($conn, $user_id) {
    $language = 'english'; // 默认语言
    
    // 查询用户设置
    $sql = "SELECT language FROM users_settings WHERE employee_id = ?";
    $stmt = $conn->prepare($sql);
    
    if ($stmt) {
        $stmt->bind_param("i", $user_id);
        $stmt->execute();
        $result = $stmt->get_result();
        
        if ($result && $result->num_rows > 0) {
            $row = $result->fetch_assoc();
            $language = $row['language'];
        }
        
        $stmt->close();
    }
    
    return $language;
}

// 如果用户已登录，获取其语言设置
$language = 'english'; // 默认语言
if (isset($_SESSION['loggedin']) && $_SESSION['loggedin'] === true && isset($_SESSION['id'])) {
    $language = getUserLanguage($conn, $_SESSION['id']);
}

// 加载语言文件
$lang_file = 'lang/en.php'; // 默认英语
if ($language == 'chinese') {
    $lang_file = 'lang/zh.php';
}

// 包含语言文件
include_once $lang_file;

// 如果$lang未定义，使用默认值
if (!isset($lang) || !is_array($lang)) {
    $lang = [];
}

// 翻译函数
function __($key) {
    global $lang;
    return isset($lang[$key]) ? $lang[$key] : $key;
}
?>

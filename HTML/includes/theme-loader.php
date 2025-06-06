<?php
// 主题加载器 - 在所有页面中包含

// 获取当前用户的主题设置
function getUserTheme($conn, $user_id) {
    $theme = 'raspberry_pi'; // 默认主题
    
    // 查询用户设置
    $sql = "SELECT theme FROM users_settings WHERE employee_id = ?";
    $stmt = $conn->prepare($sql);
    
    if ($stmt) {
        $stmt->bind_param("i", $user_id);
        $stmt->execute();
        $result = $stmt->get_result();
        
        if ($result && $result->num_rows > 0) {
            $row = $result->fetch_assoc();
            $theme = $row['theme'];
        }
        
        $stmt->close();
    }
    
    return $theme;
}

// 如果用户已登录，获取其主题设置
$theme = 'raspberry_pi'; // 默认主题
if (isset($_SESSION['loggedin']) && $_SESSION['loggedin'] === true && isset($_SESSION['id'])) {
    $theme = getUserTheme($conn, $_SESSION['id']);
}
?>

<!-- 主题相关CSS和JavaScript -->
<link rel="stylesheet" href="css/themes.css">
<script>
    // 将PHP变量传递给JavaScript
    const phpTheme = '<?php echo $theme; ?>';
</script>
<script src="js/theme-loader.js"></script>
